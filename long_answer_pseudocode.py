
# Schematic pseudocode for hitting projections and player valuation.
# All historical estimates respect their original information cutoffs.
# Fielding and baserunning projections are assumed to be provided
# by separate, existing models.

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
# 2. ESTIMATE LATENT HITTING SKILLS
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

    # Take fielding and baserunning projections as given
    # from existing R&D projection systems. For historical
    # validation, use appropriately point-in-time estimates.
    other_projections = get_existing_projections_as_of(
        data, cutoff,
        components=["fielding", "baserunning"],
        seasons=M
    )

    # Maintain distinct playing-time / deployment scenarios.
    results = {
        "optimized": [],
        "empirical": []
    }

    for sim in range(N):

        # ----------------------------------------------------
        # 4A. SAMPLE HITTING SKILLS AND FUTURE TRAJECTORIES
        # ----------------------------------------------------

        # Sample from the joint posterior distribution of
        # latent hitting skills produced by the SSM.
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
        # 4B. GENERATE SIMULATED BATTING LINES
        # ----------------------------------------------------

        # Each sampled hitting skill trajectory produces a
        # corresponding batting line for each future season.
        # These are PA outcome rates/probabilities rather than
        # accrued counting statistics.
        simulated_batting_lines = []

        for year in range(M):

            skills = skill_paths[year]
            environment = environments[year]

            # Map simulated hitting skills into hierarchical
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

            batting_line = batting_line_from_pa_probabilities(
                p_pa, outcomes=PA_OUTCOMES
            )

            simulated_batting_lines.append(batting_line)

        # ----------------------------------------------------
        # 4C. MODEL PLAYING TIME AND ACCRUED PRODUCTION
        # ----------------------------------------------------

        # Both approaches use the same simulated batting lines,
        # but determine playing time through different methods.
        for strategy in ["optimized", "empirical"]:

            roster = initialize_roster_state(history)
            contract_state = initialize_contract_state(history)
            annual_surplus = []
            annual_results = []

            for year in range(M):

                skills = skill_paths[year]
                environment = environments[year]
                batting_line = simulated_batting_lines[year]

                # Use fielding and baserunning projections from
                # existing models; their construction is outside
                # the scope of this hitting projection system.
                fielding_rates = other_projections.fielding[year]
                baserunning_rv_per_pa = (
                    other_projections.baserunning[year]
                )

                # Expected hitting value per PA from this
                # particular simulated batting line.
                hitting_rv_per_pa = expected_hitting_run_value(
                    batting_line, environment.linear_weights
                )

                # Include baserunning in total offensive value,
                # avoiding double-counting across components.
                total_offensive_rv_per_pa = (
                    hitting_rv_per_pa + baserunning_rv_per_pa
                )

                # Simulate health, promotion, and eligibility
                # conditional on the evolving player states.
                availability = simulate_player_availability(
                    skills, roster, contract_state, year
                )

                if strategy == "optimized":

                    # Optimize deployment across the roster using
                    # projected hitting, baserunning, and fielding
                    # values, subject to roster constraints.
                    deployment = solve_roster_MILP(
                        total_offensive_rv_per_pa,
                        fielding_rates,
                        positional_flexibility,
                        availability,
                        roster_constraints
                    )

                else:

                    # Empirical approach incorporates the
                    # simulated batting line and existing
                    # non-hitting projections, alongside
                    # incumbency, contracts, and availability.
                    expected_deployment = (
                        model.playing_time_model.predict(
                            batting_line,
                            fielding_rates,
                            baserunning_rv_per_pa,
                            roster,
                            contract_state,
                            availability
                        )
                    )

                    deployment = enforce_roster_constraints(
                        expected_deployment, availability
                    )

                # Deployment determines both plate appearances
                # and defensive innings by position.
                playing_time = deployment.plate_appearances
                defensive_innings = deployment.innings_by_position

                # Accrue production from the ALREADY simulated
                # batting line and existing non-hitting projections.
                # No additional batting-line draw is performed.
                hitting_runs = apply_playing_time(
                    batting_line,
                    playing_time,
                    environment.linear_weights
                )

                baserunning_runs = (
                    baserunning_rv_per_pa * playing_time
                )

                fielding_runs = accrue_fielding_value(
                    fielding_rates,
                    defensive_innings
                )

                accrued_production = {
                    "hitting": hitting_runs,
                    "baserunning": baserunning_runs,
                    "fielding": fielding_runs
                }

                # Evaluate total marginal wins in roster context.
                marginal_wins = calculate_marginal_roster_wins(
                    accrued_production,
                    deployment,
                    roster,
                    environment
                )

                # ------------------------------------------------
                # 4D. SIMULATE CONTRACTS AND PLAYER VALUE
                # ------------------------------------------------

                # Contract outcomes depend on the production
                # and playing time of this particular simulation.
                costs = model.contract_models.simulate(
                    marginal_wins,
                    accrued_production,
                    deployment,
                    contract_state,
                    year
                )

                player_value = financial_value(
                    marginal_wins, organization_context
                )
                annual_surplus.append(player_value - costs)

                annual_results.append({
                    "batting_line": batting_line,
                    "deployment": deployment,
                    "production": accrued_production,
                    "marginal_wins": marginal_wins,
                    "costs": costs
                })

                # Update roster, service time, and contract
                # status before simulating the following season.
                roster, contract_state = update_states(
                    roster, contract_state,
                    deployment, marginal_wins, costs
                )

            # Calculate surplus and asset value within each
            # complete simulation, preserving nonlinearities.
            sv = discounted_sum(annual_surplus)

            av = calculate_asset_value(
                sv, contract_state,
                roster_context, organization_utility
            )

            results[strategy].append({
                "annual_results": annual_results,
                "surplus_value": sv,
                "asset_value": av
            })

    # Aggregate across simulations only after each player's
    # production, deployment, contracts, and value are determined.
    return summarize_player_distributions(results)


# ------------------------------------------------------------
# 5. HISTORICAL OUT-OF-SAMPLE VALIDATION
# ------------------------------------------------------------

for cutoff in chronological_holdout_dates:

    # Refit the stack using only data available by cutoff.
    # Historical training targets must also be known by cutoff.
    # External non-hitting projections must respect the same
    # historical information cutoffs.
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
