
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

        # Sample joint skill trajectories, maintaining relevant
        # correlations and propagating skill uncertainty.
        skill_paths = simulate_correlated_skill_paths(
            model.current_skills,
            model.aging,
            seasons=M
        )

        # Current-season environment incorporates only
        # information available at cutoff; future seasons
        # have projected environments and linear weights.
        environments = simulate_run_environment_path(
            history, seasons=M, shrink_to_prior=True
        )

        for strategy in ["optimized", "empirical"]:

            roster = initialize_roster_state(history)
            contract_state = initialize_contract_state(history)
            annual_surplus = []

            for year in range(M):

                skills = skill_paths[year]
                environment = environments[year]

                # Hierarchical PA outcome probabilities.
                p_root = model.outcome_model.predict_root(
                    skills, environment
                )
                p_bip = model.outcome_model.predict_bip(
                    skills, environment
                )
                p_pa = combine_conditional_probabilities(
                    p_root, p_bip
                )

                # Expected offensive contributions per PA.
                rv_per_pa = expected_run_value(
                    p_pa, environment.linear_weights
                )

                # Simulate health, promotion, and eligibility
                # conditional on the evolving player states.
                availability = simulate_player_availability(
                    skills, roster, contract_state, year
                )

                if strategy == "optimized":

                    # Optimize deployment across the whole roster.
                    playing_time = solve_roster_MILP(
                        rv_per_pa,
                        projected_defense,
                        positional_flexibility,
                        availability,
                        roster_constraints
                    )

                else:

                    # Empirical approach incorporates historical
                    # playing time, incumbency, contracts, etc.
                    expected_pt = model.playing_time_model.predict(
                        skills, roster, contract_state,
                        availability
                    )
                    playing_time = enforce_roster_constraints(
                        expected_pt, availability
                    )

                # Simulate realized batting lines conditional on
                # PA probabilities and assigned playing time.
                batting_lines = sample_batting_lines(
                    p_pa, playing_time
                )

                # Evaluate value in roster context, including
                # appropriate non-hitting contributions.
                marginal_wins = calculate_marginal_roster_wins(
                    batting_lines, playing_time,
                    projected_defense, roster
                )

                # Simulate contract costs conditional on realized
                # production, service time, and contract status.
                costs = model.contract_models.simulate(
                    marginal_wins, contract_state, year
                )

                player_value = financial_value(
                    marginal_wins, organization_context
                )
                annual_surplus.append(player_value - costs)

                # Update roster, service time, and contract state.
                roster, contract_state = update_states(
                    roster, contract_state,
                    playing_time, marginal_wins, costs
                )

            # Calculate value within each simulation before
            # aggregating, preserving nonlinear effects.
            sv = discounted_sum(annual_surplus)

            av = calculate_asset_value(
                sv, contract_state,
                roster_context, organization_utility
            )

            results[strategy].append({
                "surplus_value": sv,
                "asset_value": av
            })

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
