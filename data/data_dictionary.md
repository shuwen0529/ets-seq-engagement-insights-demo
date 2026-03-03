## synthetic_users.csv
- user_id: unique user id
- segment: high/mid/low engagement propensity
- platform_pref: web/mobile preference
- baseline_motivation: numeric latent propensity
- outcome_label: binary target outcome (e.g., completion / improvement)

## synthetic_events.csv
- user_id: links to synthetic_users
- event_timestamp: timestamp of event
- event_type: one of {practice_session, assessment_attempt, score_feedback, content_review, re_engage}
- platform: web/mobile
- score: numeric score for assessment_attempt, otherwise NA
- segment: copied from user profile for convenience
