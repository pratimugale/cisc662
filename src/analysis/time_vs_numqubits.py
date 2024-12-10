import matplotlib.pyplot as plt
import pandas as pd


# draw individual plots of qubits vs time
filenames = ['delta.csv', 'bridges-2.csv', 'cpu.csv']
for filename in filenames:
    df = pd.read_csv(f'results/{filename}')
    df = df[df['Optimizer'] == 'COBYLA']
    df = df[df['NumberLayers'] <= 3]

    grp = df.groupby(by=["NumberLayers"])

    for layer, groups in grp:
        print(groups)
        plt.plot(groups['NumberNodes'], groups['TimeOptimize'], label=str(layer[0]))

    plt.xlabel('Number of Qubits')
    plt.ylabel('Time taken for Optimization')
    plt.title('Optimization Time vs Number of Qubits')

    # Show legend
    plt.legend(title='Number of Layers')
    plt.show()

filenames = ['delta.csv', 'bridges-2.csv', 'cpu.csv']

dfDelta = pd.read_csv(f'results/delta.csv')
dfBridges2 = pd.read_csv(f'results/bridges-2.csv')
dfCPU = pd.read_csv(f'results/cpu.csv')

dfDelta = dfDelta[dfDelta['Optimizer'] == 'COBYLA']
dfBridges2 = dfBridges2[dfBridges2['Optimizer'] == 'COBYLA']
dfCPU = dfCPU[dfCPU['Optimizer'] == 'COBYLA']

dfDelta = dfDelta[dfDelta['NumberLayers'] == 1]
dfBridges2 = dfBridges2[dfBridges2['NumberLayers'] == 1]
dfCPU = dfCPU[dfCPU['NumberLayers'] == 1]

fig = plt.figure()
ax = plt.subplot(111)
dfDelta.plot(x='NumberNodes', y='TimeOptimize', kind='line', ax=ax, label='delta')
dfBridges2.plot(x='NumberNodes', y='TimeOptimize', kind='line', ax=ax, label='bridges-2')
dfCPU.plot(x='NumberNodes', y='TimeOptimize', kind='line', ax=ax, label='cpu')

# dfDelta.plot(x='NumberNodes', y='TimeOptimize', kind='line')
# dfBridges2.plot(x='NumberNodes', y='TimeOptimize', kind='line')
# dfCPU.plot(x='NumberNodes', y='TimeOptimize', kind='line')

plt.xlabel('Number of Qubits')
plt.ylabel('Time taken for Optimization')
plt.title('Optimization Time vs Number of Qubits')
# Show legend
plt.legend(title='Machine')
plt.show()



dtype = {"MaxCutPartition": str}
df = pd.read_csv('results/delta.csv')
df = df[df['Optimizer'] == 'COBYLA']
df = df[df['NumberLayers'] <= 10]

grp = df.groupby(by=["NumberNodes"])

for layer, groups in grp:
    print(groups)
    plt.plot(groups['NumberLayers'], groups['TimeOptimize'], label=str(layer[0]))
#  for i, row in groups.iterrows():  # i will be index, row a pandas Series

plt.xlabel('Number of Layers')
plt.ylabel('Time taken for Optimization')
plt.title('Optimization Time vs Number of Layers')

# Show legend
plt.legend()
plt.show()




df = pd.read_csv('results/delta.csv')
df = df[df['Optimizer'] == 'COBYLA']
df = df[df['NumberLayers'] <= 10]

grp = df.groupby(by=["NumberLayers"])


for layer, groups in grp:
    print(groups)
    plt.plot(groups['NumberNodes'], groups['TimeOptimize'], label=str(layer[0]))
#  for i, row in groups.iterrows():  # i will be index, row a pandas Series

plt.xlabel('Number of Qubits')
plt.ylabel('Time taken for Optimization')
plt.title('Optimization Time vs Number of Qubits')

# Show legend
plt.legend()
plt.show()