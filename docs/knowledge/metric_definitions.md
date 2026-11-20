# GameOps AI - metric definitions

## Positive and negative review percentage
Positive % = reviews with a positive recommendation (voted_up = true) / all included reviews in the filter window.
Negative % = 100 - positive %. These describe the collected reviews only, not every player of the game.
A 95% Wilson confidence interval is reported for shares, and windows with fewer than 30 reviews are flagged as small samples.

## Complaint category share
Complaint share = reviews predicted to contain the category / base reviews. The default base is negative reviews; the "all" scope uses every review.
A review can belong to several categories (multi-label). Categories are model predictions from the selected complaint classifier, not verified labels.

## Period comparison
compare_feedback_trends reports review counts, negative share and category shares for two periods, the change in percentage points, and a two-proportion z-test p-value.
A significant change is an association in time. It does not prove that a game update or any other event caused it.

## Emerging issue
A category whose share in the most recent 28 days is higher than in the previous 28 days with a two-proportion test p-value below 0.05.

## Review volume
Number of included reviews per day, week or month, using the review creation timestamp in UTC.
