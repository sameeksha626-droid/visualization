# Data Visualization with Matplotlib & Seaborn

Project 02 - Data Science & AI (InternCircle)

Charts built from seaborn's `tips` dataset (244 restaurant meals).

## Questions
1. How are bills distributed? -> histogram
2. Do tips differ by day? -> box plot
3. Does party size change the average tip? -> bar chart
4. Does a bigger bill mean a bigger tip? -> scatter plot
5. Which numbers move together? -> correlation heatmap

## Files
- `tips_visualization.ipynb` - the notebook (run top to bottom)
- `tips_visualization.py` - same code as a script
- `charts/` - saved charts (created when you run the notebook), including `tips_overview.png`, the 2x2 dashboard

## How to run
`pip install pandas numpy matplotlib seaborn jupyter`, then open the notebook and Run All
(or use Google Colab).

## Takeaways
1. 70% of bills are between $10 and $25; a few go above $40.
2. Median tips are similar across days ($2.30 to $3.15); Sun is highest.
3. Average tip rises with party size, from $1.44 (1 person) to $5.22 (6 people).
4. Tip grows with the bill (correlation r = 0.68); lunch and dinner follow the same pattern.
5. The strongest relationship is total_bill and tip (r = 0.68).
