# this script is used to create a network using adtech data

#%% import libraries
#=======================================================================================================================
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import datetime
from shapely.geometry import Point
from geopy.distance import geodesic
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon

from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from itertools import combinations
import matplotlib.patches as mpatches
#=======================================================================================================================

os.getcwd()
# os.chdir('adtech/experiment')
#=======================================================================================================================

#%% Generate a large and complex dummy dataset
#=======================================================================================================================
np.random.seed(42)

num_devices = 15  # Adjust this as needed
all_device_ids = []
all_latitudes = []
all_longitudes = []
all_date_times = []
all_obs_per_device = []
all_device_risk_status = []
all_most_common_app_used = []
all_home_city = []

# Sample data for new variables
apps = ['Grindr', 'GoogleMaps', 'Hornet', 'Deliveroo']
cities = ['Newcastle', 'London', 'Manchester', 'Glasgow']

# Generate a risk status and other attributes for each device
device_risk_statuses = np.random.choice(['low', 'medium', 'high'], num_devices)
device_apps = np.random.choice(apps, num_devices)
device_cities = np.random.choice(cities, num_devices)

for i in range(num_devices):
    num_observations_per_device = np.random.randint(35, 60)
    device_ids = np.repeat(i, num_observations_per_device)
    np.random.seed(i)
    latitudes = np.random.uniform(48.87, 48.90, num_observations_per_device)
    longitudes = np.random.uniform(2.37, 2.40, num_observations_per_device)
    date_times = pd.date_range(start='2024-03-01 01:00:00', periods=num_observations_per_device, freq='30min')
    obs_per_device = np.repeat(num_observations_per_device, num_observations_per_device)
    # use the predefined risk status for the device
    risk_status = np.repeat(device_risk_statuses[i], num_observations_per_device)
    most_common_app_used = np.repeat(device_apps[i], num_observations_per_device)
    home_city = np.repeat(device_cities[i], num_observations_per_device)

    all_device_ids.extend(device_ids)
    all_latitudes.extend(latitudes)
    all_longitudes.extend(longitudes)
    all_date_times.extend(date_times)
    all_obs_per_device.extend(obs_per_device)
    all_device_risk_status.extend(risk_status)
    all_most_common_app_used.extend(most_common_app_used)
    all_home_city.extend(home_city)

data = {
    'device_id': all_device_ids,
    'lat': all_latitudes,
    'lon': all_longitudes,
    'date_time': all_date_times,
    'obs_per_device': all_obs_per_device,
    'device_risk_status': all_device_risk_status,
    'most_common_app_used': all_most_common_app_used,
    'home_city': all_home_city
}

paris_df1 = pd.DataFrame(data)

#%% convert it into a geo dataframe
#=======================================================================================================================
geometry = [Point(xy) for xy in zip(paris_df1['lon'], paris_df1['lat'])]
gdf = gpd.GeoDataFrame(paris_df1, geometry=geometry)

#%% TO BEGIN, WE'LL BUILD A NETWORK OF DEVICES CO-LOCATED IN SPACE AND TIME
#=======================================================================================================================
# Function to calculate distance in meters between two lat/lon points
def haversine_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).meters

# Convert coordinates to numpy array
coords = paris_df1[['lat', 'lon']].to_numpy()

#%% Use DBSCAN clustering to find colocations within 100 meters
#=======================================================================================================================
db = DBSCAN(eps=500/6371000, min_samples=2, metric='haversine').fit(np.radians(coords))
paris_df1['cluster'] = db.labels_

#%% get cluster size
#=======================================================================================================================
paris_df1['unique_devices_per_cluster'] = paris_df1.groupby('cluster')['device_id'].transform('nunique')
# ungroup
paris_df1 = paris_df1.reset_index(drop=True)

# Filter out noise where only 1 row per cluster (label -1) and clusters with only 1 unique device
#=======================================================================================================================
paris_df1 = paris_df1[ (paris_df1['cluster'] != -1) & (paris_df1['unique_devices_per_cluster'] > 1) ]

#%% now let's create a day vs night variable
#=======================================================================================================================
# extract an hour variable
paris_df1['hour'] = paris_df1['date_time'].dt.hour
paris_df1['day_night'] = np.where((paris_df1['date_time'].dt.hour >= 6) & (paris_df1['date_time'].dt.hour < 18), 'day', 'night')

#%% now let's create a new rounded coordinate variable using geohash
#=======================================================================================================================
import geohash2
def geohash_coordinates(lat, lon, precision):
    return geohash2.encode(lat, lon, precision)

# Create a new column for the geohash
paris_df1['geohash_7'] = paris_df1.apply(lambda x: geohash_coordinates(x['lat'], x['lon'], precision=6), axis=1)
paris_df1['geohash_7_count'] = paris_df1.groupby(['device_id','geohash_7'])['geohash_7'].transform('count')
paris_df1 = paris_df1.sort_values(by=['device_id', 'geohash_7_count'], ascending=False)

#%% Create an empty graph
#=======================================================================================================================
G = nx.Graph()

#%% Build a network of devices with certain characteristics
# in the complete version, there should be:
# 'areas' (polygons) that can act as filters; time based filters; etc.
# what are the features of suspicious devices (travel, ports, high nighttime activity)?

# key point: there are two ways to do this:
# 1) you filter the df by devices of interest and network with each other
# 2) for devices of interest, you network across the entire dataframe (computionally costly) - now applied to the function

def haversine_distance(lat1, lon1, lat2, lon2):
    # Placeholder for the actual haversine distance calculation
    # Replace with the actual implementation
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5

# This function should give you three options:
# 1) just build networks within the subset (e.g., exclude family)
# 2) build networks for those within the subset, but include their edges beyond it
# 3)

def create_network(df, G, day_night=None, home_city=None, apps_used=None, space_threshold_metres=100,
                   time_threshold_minutes=180, area_of_interest=None, device_ids_of_interest=None, network_scope=None):
    # Create a GeoDataFrame from the DataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))

    # Convert area_of_interest to a Polygon if it is a string of coordinates
    if area_of_interest:
        area_of_interest = Polygon(area_of_interest)

    # Lambda functions to handle default values
    day_night = day_night if day_night else df['day_night'].unique()
    home_city = home_city if home_city else df['home_city'].unique()
    apps_used = apps_used if apps_used else df['most_common_app_used'].unique()

    # Filter the data
    filter_df = gdf[gdf['day_night'].isin(day_night)]
    filter_df = filter_df[filter_df['home_city'].isin(home_city)]
    filter_df = filter_df[filter_df['most_common_app_used'].isin(apps_used)]

    # Filter based on area of interest
    if area_of_interest:
        filter_df = filter_df[filter_df.within(area_of_interest)]

    # If network_scope is 'Narrow', restrict the data to the filtered DataFrame
    # if network_scope[0] == 'Narrow': # note, if passed as a list, must extract the first element, so pass as a string below
    if network_scope == 'Narrow':  # note, if passed as a list, must extract the first element, so pass as a string below
        gdf = filter_df
    else:
        gdf = gdf.copy()

    # Determine the device_ids_of_interest based on the filtered DataFrame
    if device_ids_of_interest is None:
        device_ids_of_interest = filter_df['device_id'].unique()
    else:
        device_ids_of_interest = filter_df[filter_df['device_id'].isin(device_ids_of_interest)]['device_id'].unique()

    # Get networks for a subset of devices, checking their connections across the entire dataset
    devices_of_interest_df = gdf[gdf['device_id'].isin(device_ids_of_interest)]

    # Construct the network
    for i, row1 in devices_of_interest_df.iterrows():
        for j, row2 in gdf.iterrows():
            if row1['device_id'] == row2['device_id']:
                continue

            # Calculate the spatial distance between the two devices
            spatial_distance = haversine_distance(row1['lat'], row1['lon'], row2['lat'], row2['lon'])

            # Calculate the temporal distance between the two devices
            temporal_distance = abs((row1['date_time'] - row2['date_time']).total_seconds()) / 60  # convert to minutes

            # Check if both spatial and temporal distances meet the criteria
            if spatial_distance <= space_threshold_metres and temporal_distance <= time_threshold_minutes:
                # Add edge only if the source device (row1) is in devices_of_interest
                if row1['device_id'] in device_ids_of_interest:
                    # If an edge already exists, increment the weight
                    if G.has_edge(row1['device_id'], row2['device_id']):
                        G[row1['device_id']][row2['device_id']]['weight'] += 1
                    else:
                        # Otherwise, add a new edge with an initial weight of 1
                        G.add_edge(row1['device_id'], row2['device_id'], weight=1)

    # Filter the edges to only include those where device_id_1 is in devices_of_interest
    edges_to_remove = [(u, v) for u, v in G.edges if u not in device_ids_of_interest]
    G.remove_edges_from(edges_to_remove)

    return G

# Example usage of the function
# Initialize the graph
G = nx.Graph()

#=======================================================================================================================
# Assuming paris_df1 is your DataFrame with necessary columns
# Build the network for specific devices of interest, connecting them to any other device in the entire DataFrame
G = create_network(paris_df1, G, day_night=['day', 'night'], space_threshold_metres=100, time_threshold_minutes=180,
                   apps_used = ['Grindr'],
                   # device_ids_of_interest=[8, 12],
                   network_scope = 'Broad')

#%% NOTE: IT MAY MAKE SENSE TO DO THE BELOW IN A RESTRICTED NETWORK (E.G., ONLY HIGH RISK DEVICES IN FOCAL'S NETWORK)
# TO ACHIEVE THIS, WE CAN TURN IT INTO A FUNCTION THAT CAN BE APPLIED TO EACH DIFFERENT NETWORK

#%% Convert the graph G to a DataFrame for visualization
#=======================================================================================================================
graph_data = []

#%% Iterate over the edges in the graph to extract the data
#=======================================================================================================================
graph_object = G

for u, v, data in graph_object.edges(data=True):
    graph_data.append([u, v, data['weight']])
# Create a DataFrame from the graph data
graph_df = pd.DataFrame(graph_data, columns=['device_id_1', 'device_id_2', 'weight'])
# remove self connections
graph_df = graph_df[graph_df['device_id_1'] != graph_df['device_id_2']]
# let's get the size of each device's network
graph_df['focal_number_unique_connections'] = graph_df.groupby('device_id_1')['device_id_1'].transform('count')
graph_df['focal_total_interactions'] = graph_df.groupby('device_id_1')['weight'].transform('sum') # weights are pairwise edges

#%% merge with the risk status for device_id_2
#=======================================================================================================================
merge_data = graph_df.copy()
# rename as device_id
merge_data = merge_data.rename(columns={'device_id_1': 'device_id_focal', 'device_id_2': 'device_id'})
# now merge with the paris_df1 dataframe to get original variables
merge_data = merge_data.merge(paris_df1[['device_id', 'device_risk_status']], on = 'device_id', how='left')
# drop duplicates
merge_data = merge_data.drop_duplicates(subset = ['device_id_focal', 'device_id'])
# string to numeric dictionary
risk_status_dict_1 = {'low': 1, 'medium': 2, 'high': 3}
# map the risk status to numeric
merge_data['device_risk_status'] = merge_data['device_risk_status'].map(risk_status_dict_1)

#%% measure centrality
#=======================================================================================================================
degree_centrality = nx.degree_centrality(G)
# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
# Calculate closeness centrality
closeness_centrality = nx.closeness_centrality(G, distance='weight')
# Calculate eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
# Calculate PageRank
pagerank = nx.pagerank(G, weight='weight')
# Calculate HITS scores
hubs, authorities = nx.hits(G, max_iter=500)

#%% Create a DataFrame for centrality measures
centrality_df = pd.DataFrame({
    'device_id_focal': list(G.nodes),
    'degree_centrality': [degree_centrality[node] for node in G.nodes],
    'betweenness_centrality': [betweenness_centrality[node] for node in G.nodes],
    'closeness_centrality': [closeness_centrality[node] for node in G.nodes],
    'eigenvector_centrality': [eigenvector_centrality[node] for node in G.nodes],
    'pagerank': [pagerank[node] for node in G.nodes],
    'hubs': [hubs[node] for node in G.nodes],
    'authorities': [authorities[node] for node in G.nodes]
})

#%% Merge with the existing graph DataFrame
#=======================================================================================================================
graph_df_centrality = merge_data.merge(centrality_df, on ='device_id_focal',how='left')

#%% next, we'll create a risk weighted network score that can be used to identify high risk devices
# I mean that a device's centrality (colour?) will be weighted by the risk status of the devices it interacts with


#%% let's get corr between each pair of variables using matrix
# Calculate the correlation matrix
correlation_matrix = graph_df_centrality.corr()
# Set the figure size before plotting
plt.figure(figsize=(16, 14))
# Plot the heatmap
sns.heatmap(correlation_matrix.iloc[[3, 4, 6, 7, 8, 9, 10, 11, 12], :], annot=True, cmap='coolwarm', fmt=".2f", cbar=True, square=True, linewidths=0.5, annot_kws={"size": 12})
# Add a title
plt.title('Correlation Matrix of Network Measures')
# angle the x-axis labels for better readability
plt.xticks(rotation=45)
plt.yticks(rotation=45)
# Show the plot
plt.show()

#%% risk score (e.g., total interactions with high risk devices)

#%% LET'S PLOT THE NETWORK WITH FILTERING OPTIONS
import matplotlib.patches as mpatches
#%% NOTE: THESE FILTERS MAY REMOVE SOME EDGE NUMBERS AS WELL AS NODES
def calculate_centrality_measures(G):
    # Calculate various centrality measures
    centrality_measures = {
        'degree_centrality': nx.degree_centrality(G),
        'betweenness_centrality': nx.betweenness_centrality(G, weight='weight'),
        'closeness_centrality': nx.closeness_centrality(G, distance='weight'),
        'eigenvector_centrality': nx.eigenvector_centrality(G, weight='weight'),
        'pagerank': nx.pagerank(G, weight='weight'),
        'hubs': nx.hits(G, max_iter=500)[0],
        'authorities': nx.hits(G, max_iter=500)[1]
    }
    return centrality_measures

def plot_filterer_network(G, df, device_ids_of_interest = None, app_list=None, city_list=None, risk_status_list=None, day_night=None, size_by='degree_centrality'):
    # Create a copy of the dataframe to work with
    filtered_df = df.copy()

    # Filter the data based on the provided lists
    # if device_ids_of_interest:
    #     filtered_df = filtered_df[filtered_df['device_id'].isin(device_ids_of_interest)]
    if app_list:
        filtered_df = filtered_df[filtered_df['most_common_app_used'].isin(app_list)]
    if city_list:
        filtered_df = filtered_df[filtered_df['home_city'].isin(city_list)]
    if risk_status_list:
        filtered_df = filtered_df[filtered_df['device_risk_status'].isin(risk_status_list)]
    if day_night:
        filtered_df = filtered_df[filtered_df['day_night'].isin(day_night)]

    # Calculate centrality measures
    centrality_measures = calculate_centrality_measures(G)

    # Ensure the centrality measure exists
    if size_by not in centrality_measures:
        raise ValueError(f"Invalid centrality measure: {size_by}")

    centrality = centrality_measures[size_by]

    # Filter the graph based on the filtered devices
    # filtered_G = G.subgraph(filtered_df['device_id'].unique())
    filtered_G = G.subgraph(filtered_df[device_ids_of_interest])


    # Remove self-loops
    filtered_G = nx.Graph(filtered_G)  # Create a new graph to avoid modifying the original
    filtered_G.remove_edges_from(nx.selfloop_edges(filtered_G))

    # Set node sizes based on the chosen centrality measure, scaled for visibility
    node_sizes = [centrality.get(node, 0) * 5000 for node in filtered_G.nodes()]  # Scale by 1000 for visibility

    # Set edge widths proportional to the edge weights
    edge_weights = [filtered_G[u][v].get('weight', 1) for u, v in filtered_G.edges()]  # Default weight to 1 if not found

    # Define a color mapping for risk statuses
    risk_color_map = {
        'low': 'green',
        'medium': 'orange',
        'high': 'red'
    }

    # Ensure the filtered_df has the required columns
    assert 'device_id' in filtered_df.columns
    assert 'device_risk_status' in filtered_df.columns

    # Set node colors based on the risk status
    risk_status_dict = filtered_df.set_index('device_id')['device_risk_status'].to_dict()
    node_colors = [risk_color_map[risk_status_dict.get(node, 'low')] for node in filtered_G.nodes()]  # Default to 'low' risk if not found

    # Use the positions from the original graph for consistency
    pos = nx.spring_layout(filtered_G, seed=42)  # Layout for our nodes

    # Draw the graph
    plt.figure(figsize=(12, 10))

    # Generate the filter list for the title
    filter_list = []
    if app_list:
        filter_list.extend(app_list)
    if city_list:
        filter_list.extend(city_list)
    if risk_status_list:
        filter_list.extend(risk_status_list)

    title_str = 'Network of Devices Colocated in Space and Time'
    if filter_list:
        title_str += f"\nSized by {size_by.capitalize()}, Filtered by: " + ', '.join(filter_list)
    else:
        title_str += f"\nSized by {size_by.capitalize()}"

    plt.title(title_str, fontsize=20, fontweight='bold', color='black', loc='center', pad=30)

    # Create the color key
    low_patch = mpatches.Patch(color='green', label='Low Risk')
    medium_patch = mpatches.Patch(color='orange', label='Medium Risk')
    high_patch = mpatches.Patch(color='red', label='High Risk')
    plt.legend(handles=[low_patch, medium_patch, high_patch], loc='best', fontsize=16)

    nx.draw(filtered_G, pos=pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color='gray',
            width=edge_weights)

    # Display the plot
    # save with size and filters in name
    # plt.savefig('plots/network_plot.png', dpi = 200)
    file_name = f"{size_by}_{'_'.join(filter_list)}.png"
    plt.savefig(file_name, dpi=200)

    plt.show()

#%% example unfiltered
plot_filterer_network(G, paris_df1, device_ids_of_interest= [13], size_by='pagerank')

#%% Example usage: Filter the network by risk status
plot_filterer_network(G, paris_df1, risk_status_list=['high', 'medium'])

#%% grindr users
plot_filterer_network(G, paris_df1, app_list=['Grindr', 'Deliveroo'])

#%% nighttime
plot_filterer_network(G, paris_df1, day_night=['night'])

#%% Function to get a subgraph for a specific device
def get_device_subgraph(G, device_id):
    # Get the neighbors of the specified device
    neighbors = list(G.neighbors(device_id))

    # Create a subgraph containing only the focal device and its neighbors
    nodes_to_include = [device_id] + neighbors
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in nodes_to_include)

    # Add edges only from the specified device to its neighbors
    if SG.is_multigraph():
        SG.add_edges_from(
            (device_id, nbr, key, G[device_id][nbr][key])
            for nbr in neighbors
            for key in G[device_id][nbr]
        )
    else:
        SG.add_edges_from(
            (device_id, nbr, G[device_id][nbr])
            for nbr in neighbors
        )

    SG.graph.update(G.graph)
    return SG

# get each node/device's risk status, all_most_common_app_used, and home_city
risk_status_dict = paris_df1.set_index('device_id')['device_risk_status'].to_dict()
most_common_app_used_dict = paris_df1.set_index('device_id')['most_common_app_used'].to_dict()
home_city_dict = paris_df1.set_index('device_id')['home_city'].to_dict()
number_of_observations_dict = paris_df1.set_index('device_id')['obs_per_device'].to_dict()
# get number of other nodes in the network
number_of_neighbors_dict = paris_df1.groupby('device_id')['cluster'].transform('nunique').to_dict()

nx.set_node_attributes(G, risk_status_dict, 'risk_status')
nx.set_node_attributes(G, most_common_app_used_dict, 'most_common_app_used')
nx.set_node_attributes(G, home_city_dict, 'home_city')
nx.set_node_attributes(G, number_of_observations_dict, 'number_of_observations')
nx.set_node_attributes(G, number_of_neighbors_dict, 'number_of_neighbors')

#%% create a plotting function for a device
from matplotlib.offsetbox import AnchoredText

#%% Function to plot the subgraph and add an annotation box
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText

def plot_device_subgraph(G, device_id, df, app_list=[], city_list=[], risk_status_list=[]):
    # Get the subgraph for the specified device
    SG = get_device_subgraph(G, device_id)

    # Create a copy of the dataframe to work with
    filtered_df = df.copy()

    # Filter the data based on the provided lists
    if app_list:
        filtered_df = filtered_df[filtered_df['most_common_app_used'].isin(app_list)]
    if city_list:
        filtered_df = filtered_df[filtered_df['home_city'].isin(city_list)]
    if risk_status_list:
        filtered_df = filtered_df[filtered_df['device_risk_status'].isin(risk_status_list)]

    # Calculate the number of observations for each device (node size)
    device_obs = filtered_df['device_id'].value_counts().to_dict()
    node_sizes = [device_obs.get(node, 1) * 10 for node in SG.nodes()]  # Scale by 10 for visibility

    # Set edge widths proportional to the edge weights
    edge_weights = [SG[u][v].get('weight', 1) for u, v in SG.edges()]  # Default weight to 1 if not found

    # Define a color mapping for risk statuses
    risk_color_map = {
        'low': 'green',
        'medium': 'orange',
        'high': 'red'
    }

    # Set node colors based on the risk status
    risk_status_dict = filtered_df.set_index('device_id')['device_risk_status'].to_dict()
    node_colors = [risk_color_map.get(risk_status_dict.get(node, 'low'), 'green') for node in
                   SG.nodes()]  # Default to 'low' risk if not found

    # Use the positions from the original graph for consistency
    pos = nx.spring_layout(SG, seed=42)  # Layout for our nodes

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Add the title before drawing
    filter_list = app_list + city_list + risk_status_list
    title_str = f"Network of Devices Colocated in Space and Time\nDevice {device_id} Subgraph"
    if filter_list:
        title_str += ', Filtered by: ' + ', '.join(filter_list)
    plt.title(title_str, fontsize=16, fontweight='bold', color='black', loc='center', pad=25)

    # Draw the subgraph
    nx.draw(SG, pos=pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color='gray',
            width=edge_weights)

    # Add the color key
    low_patch = mpatches.Patch(color='green', label='Low Risk')
    medium_patch = mpatches.Patch(color='orange', label='Medium Risk')
    high_patch = mpatches.Patch(color='red', label='High Risk')
    plt.legend(handles=[low_patch, medium_patch, high_patch], loc='best', fontsize=16)

    # Get each node/device's characteristics
    most_common_app_used_dict = filtered_df.set_index('device_id')['most_common_app_used'].to_dict()
    home_city_dict = filtered_df.set_index('device_id')['home_city'].to_dict()
    number_of_observations_dict = filtered_df['device_id'].value_counts().to_dict()

    # Get number of neighbors in the network
    neighbors = list(SG.neighbors(device_id))
    neighbors_df = filtered_df[filtered_df['device_id'].isin(neighbors)]
    number_of_neighbors = neighbors_df['device_id'].nunique()

    # Set node attributes
    nx.set_node_attributes(G, risk_status_dict, 'risk_status')
    nx.set_node_attributes(G, most_common_app_used_dict, 'most_common_app_used')
    nx.set_node_attributes(G, home_city_dict, 'home_city')
    nx.set_node_attributes(G, number_of_observations_dict, 'number_of_observations')

    # Create an annotation box with the focal device characteristics
    device_info = f"Device ID: {device_id}\n" \
                  f"Risk Status: {G.nodes[device_id]['risk_status']}\n" \
                  f"Most Common App: {G.nodes[device_id]['most_common_app_used']}\n" \
                  f"Number of Observations: {number_of_observations_dict[device_id]}\n" \
                  f"Number of Neighbors: {number_of_neighbors}\n" \
                  f"Home City: {G.nodes[device_id]['home_city']}"

    anchored_text = AnchoredText(device_info, loc='lower right', prop={'size': 14}, frameon=True, bbox_to_anchor=(1, 0), bbox_transform=plt.gcf().transFigure)
    anchored_text.patch.set_boxstyle("round,pad=0.3,rounding_size=0.2")
    plt.gca().add_artist(anchored_text)

    # save
    file_name = f"plots/device_{device_id}_subgraph_{'_'.join(filter_list)}.png"
    plt.savefig(file_name, dpi=200)

    plt.show()

# Example usage
plot_device_subgraph(G, 13, paris_df1, risk_status_list=['high', 'medium'])


#%% NOW LET'S GET A MAP OF LOCATIONS FOR DEVICE 13 FROM THE PARIS_DF1 DATAFRAME
# Get the data for device 13
from collections import Counter

device_13_data = paris_df1[paris_df1['device_id'] == 13]

# Compute travel frequencies between coordinates
coords = list(zip(device_13_data['lon'], device_13_data['lat']))
travel_paths = [(coords[i], coords[i+1]) for i in range(len(coords) - 1)]
path_counts = Counter(travel_paths)

# Plot the device locations in abstract space without a map
plt.figure(figsize=(10, 10))
plt.scatter(device_13_data['lon'], device_13_data['lat'], s=100, c='red', edgecolors='black', zorder=2)

# Plot the travel paths with line thickness based on frequency
for (start, end), count in path_counts.items():
    plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=count, zorder=1)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Device 13 Locations with Travel Paths')
# save
plt.savefig('plots/device_13_locations_travel_paths.png', dpi=200)
plt.show()

#%% TO EXPAND THIS FUNCTIONALITY, CONSIDER:
# making dot size proportional to either a) number of times at location, or b) time spent at location
# related to above, highlighting bed-down locations (e.g., where the device spends the most time)
# adding arrows to the paths to indicate directionality
# making number of arrows proportional to the speed and thickness of the path to the frequency of travel
# adding filters for specific time periods or days of the week or day/night
# layering on the paths of other devices (how to make this interpretable?), ensuring filtering by time so can see if
# travelling together
# using a heat map if the density is high
# adjusting dot number based on map resolution (more zoomed in = more dots)


#%% Function to get all connections for a specific device
def get_device_connections(graph, device_id):
    connections = []
    for neighbor in graph.neighbors(device_id):
        weight = graph[device_id][neighbor]['weight']
        connections.append([device_id, neighbor, weight])
    return connections
# Example usage: Get connections for device 1
device_id_to_query = 1
connections = get_device_connections(G, device_id_to_query)
# Convert to DataFrame for better readability
connections_df = pd.DataFrame(connections, columns=['device_id', 'connected_device_id', 'weight'])
print(f"\nConnections for device {device_id_to_query}:")
print(connections_df)


