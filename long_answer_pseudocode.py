
# Schematic pseudocode for hitting projections and player valuation.
# All historical estimates respect their original information cutoffs.

PA_OUTCOMES = ["K", "BB", "HBP", "IPO", "1B", "2B", "3B", "HR"]


# ------------------------------------------------------------
# 1. PREPARE OBSERVATIONS
# ------------------------------------------------------------

def observations_as_of(data, cutoff):
    history = data.available_as_of(cutoff)

    # Prefer historical OOS / point-in-time upstream estimates.
    # Exclude or separately assess features whose historical
    # origin cannot support an unbiased evaluation.
    upstream = get_valid_upstream_estimates(history, cutoff)

    metrics = calculate_metrics(
        history, upstream,
        metrics=[
            "PA outcomes", "80th percentile EV",
            "swing decision RV", "contact vs expected",
            "hard hit vs expected", "speed", "bat path",
            "batted ball distribution"
        ]
    )

    return adjust_for_context(
        metrics,
        level=True, competition=True,
        park=True, platoon=True
    )


# ------------------------------------------------------------
# 2. ESTIMATE LATENT SKILLS
# ------------------------------------------------------------

def estimate_skills_as_of(data, cutoff):
    # Fit SSM parameters using seasons through Y-1.
    training_end = end_of_previous_season(cutoff)

    ssm = fit_state_space_model(
        observations_as_of(data, training_end)
    )

    # Filtering may incorporate observations through cutoff,
    # but never observations occurring afterward.
    observations = observations_as_of(data, cutoff)
    filtered = kalman_filter(ssm, observations)

    return filtered.skill_posteriors_at(cutoff)


# ------------------------------------------------------------
# 3. TRAIN PROJECTION MODEL STACK
# ------------------------------------------------------------

def train_projection_stack(data, cutoff, M):

    examples = []

    for origin in historical_forecast_dates(before=cutoff):

        # Future outcomes must be observed by training cutoff.
        if not forecast_horizon_elapsed(origin, M, cutoff):
            continue

        skills = estimate_skills_as_of(data, origin)

        # Future outcomes are training targets, not predictors.
        targets = get_adjusted_future_PA_outcomes(
            data, origin, horizon=M, observed_by=cutoff
        )

        examples.append({
            "skills": skills,
            "ages": player_ages_at(origin),
            "targets": targets,
            "exposure": observed_PA_counts(targets)
        })

    # Fit constrained skill aging and hierarchical PA loading.
    aging, outcome_model = fit_joint_aging_and_outcome_model(
        examples,
        constrain_aging=True,
        correct_survivorship_bias=True,
        outcome_hierarchy={
            "PA": ["K", "BB", "HBP", "BIP"],
            "BIP": ["IPO", "1B", "2B", "3B", "HR"]
        }
    )

    # Here BIP denotes the batted-ball outcome branch,
    # including home runs.

    current_skills = estimate_skills_as_of(data, cutoff)

    # Separately fitted downstream models.
    history = data.available_as_of(cutoff)

    playing_time_model = fit_empirical_playing_time_model(history)
    contract_models = fit_contract_and_promotion_models(history)

    return ModelStack(
        current_skills, aging, outcome_model,
        playing_time_model, contract_models
    )


# ------------------------------------------------------------
# 4. SIMULATE FUTURE PRODUCTION AND PLAYER VALUE
# ------------------------------------------------------------

def simulate_player_values(model, data, cutoff, N, M):

    history = data.available_as_of(cutoff)

    # Maintain distinct playing-time / deployment scenarios.
    results = {
        "optimized": [],
        "empirical": []
    }

    for sim in range(N):

        # Sample from the joint posterior distribution of
        # latent skills produced by the SSM.
        sampled_skills = sample_joint_skill_posterior(
            model.current_skills
        )

        # Apply aging and simulate future skill trajectories,
        # maintaining relevant correlations and uncertainty.
        skill_paths = simulate_correlated_skill_paths(
            sampled_skills,
            model.aging,
            seasons=M
        )

        # Current-season environment incorporates only
        # information available at cutoff; future seasons
        # have projected environments and linear weights.
        environments = simulate_run_environment_path(
            history, seasons=M, shrink_to_prior=True
        )

        # ----------------------------------------------------
        # 4A. GENERATE SIMULATED BATTING LINES
        # ----------------------------------------------------

        # Each sampled skill trajectory produces a corresponding
        # batting line for each future season, independent of
        # playing time. These are distributions/rates per PA,
        # rather than accrued counting statistics.
        simulated_batting_lines = []

        for year in range(M):

            skills = skill_paths[year]
            environment = environments[year]

            # Map simulated latent skills into hierarchical
            # PA outcome probabilities.
            p_root = model.outcome_model.predict_root(
                skills, environment
            )
            p_bip = model.outcome_model.predict_bip(
                skills, environment
            )
            p_pa = combine_conditional_probabilities(
                p_root, p_bip
            )

            # This skill draw induces one simulated batting
            # line, represented by PA probabilities and rates.
            batting_line = batting_line_from_pa_probabilities(
                p_pa, outcomes=PA_OUTCOMES
            )

            simulated_batting_lines.append(batting_line)

        # ----------------------------------------------------
        # 4B. MODEL PLAYING TIME AND ACCRUED PRODUCTION
        # ----------------------------------------------------

        # Both approaches use the same simulated batting lines,
        # but determine playing time through different methods.
        for strategy in ["optimized", "empirical"]:

            roster = initialize_roster_state(history)
            contract_state = initialize_contract_state(history)
            annual_surplus = []

            for year in range(M):

                skills = skill_paths[year]
                environment = environments[year]
                batting_line = simulated_batting_lines[year]

                # Calculate expected offensive value per PA
                # from this particular simulated batting line.
                rv_per_pa = expected_run_value(
                    batting_line, environment.linear_weights
                )

                # Simulate health, promotion, and eligibility
                # conditional on the evolving player states.
                availability = simulate_player_availability(
                    skills, roster, contract_state, year
                )

                if strategy == "optimized":

                    # Optimize roster deployment using the
                    # simulated batting lines and constraints.
                    playing_time = solve_roster_MILP(
                        rv_per_pa,
                        projected_defense,
                        positional_flexibility,
                        availability,
                        roster_constraints
                    )

                else:

                    # Empirical approach incorporates the
                    # simulated batting line, historical playing
                    # time, incumbency, contracts, etc.
                    expected_pt = model.playing_time_model.predict(
                        batting_line, skills, roster,
                        contract_state, availability
                    )

                    playing_time = enforce_roster_constraints(
                        expected_pt, availability
                    )

                # Playing time determines accrued production
                # from the ALREADY simulated batting line.
                # No additional batting-line draw is performed.
                accrued_production = apply_playing_time(
                    batting_line, playing_time
                )

                # Evaluate accrued value in roster context,
                # including appropriate non-hitting contributions.
                marginal_wins = calculate_marginal_roster_wins(
                    accrued_production, playing_time,
                    projected_defense, roster
                )

                # ------------------------------------------------
                # 4C. SIMULATE CONTRACTS AND PLAYER VALUE
                # ------------------------------------------------

                # Contract outcomes depend on the production
                # and playing time of this particular simulation.
                costs = model.contract_models.simulate(
                    marginal_wins, contract_state, year
                )

                player_value = financial_value(
                    marginal_wins, organization_context
                )
                annual_surplus.append(player_value - costs)

                # Update roster, service time, and contract
                # status before simulating the following season.
                roster, contract_state = update_states(
                    roster, contract_state,
                    playing_time, marginal_wins, costs
                )

            # Calculate surplus and asset value within each
            # complete simulation, preserving nonlinearities.
            sv = discounted_sum(annual_surplus)

            av = calculate_asset_value(
                sv, contract_state,
                roster_context, organization_utility
            )

            results[strategy].append({
                "batting_lines": simulated_batting_lines,
                "surplus_value": sv,
                "asset_value": av
            })

    # Aggregate across simulations only after each player's
    # production, usage, and value have been determined.
    return summarize_player_distributions(results)


# ------------------------------------------------------------
# 5. HISTORICAL OUT-OF-SAMPLE VALIDATION
# ------------------------------------------------------------

for cutoff in chronological_holdout_dates:

    # Refit the stack using only data available by cutoff.
    # Historical training targets must also be known by cutoff.
    model = train_projection_stack(data, cutoff, M)

    predictions = simulate_player_values(
        model, data, cutoff, N, M
    )

    evaluate_out_of_sample(
        predictions,
        realized_outcomes_after(cutoff),
        metrics=[
            "predictive accuracy",
            "probabilistic calibration",
            "subpopulation calibration",
            "playing time accuracy",
            "valuation performance"
        ]
    )
