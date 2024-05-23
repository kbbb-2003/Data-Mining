import matplotlib.pyplot as plt
import pandas as pd

# Define the tasks and their durations
tasks = {
    'Data Collection and Preprocessing': (0, 1),
    'Model Development and Training': (1, 3),
    'Model Evaluation and Optimization': (3, 4),
    'Explainability Analysis': (4, 5),
    'Final Report Preparation': (5, 6)
}

# Create a DataFrame
df = pd.DataFrame(list(tasks.values()), columns=['Start Month', 'End Month'], index=list(tasks.keys()))

# Plotting the Gantt Chart
fig, ax = plt.subplots(figsize=(18, 10))

# Adding tasks to the chart
for i, task in enumerate(df.index):
    ax.broken_barh([(df.loc[task, 'Start Month'], df.loc[task, 'End Month'] - df.loc[task, 'Start Month'])], 
                   (i - 0.4, 0.8), facecolors=('tab:blue'))

# Formatting the chart
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df.index)
ax.set_xlabel('Months')
ax.set_ylabel('Tasks')
ax.set_title('Gantt Chart for Research Proposal')

plt.grid(True)
plt.show()