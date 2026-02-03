from caveclient import CAVEclient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from standard_transform import minnie_ds
import numpy as np
from standard_transform import minnie_ds



datastack_name = 'minnie65_public'
client = CAVEclient(datastack_name)


def conversion_three(presource, boxsource, postsource):
    # # convert to nanometers
    
    presource['pt_xt_μm'] = presource['pt_position_x'] * 4
    presource['pt_yt_μm'] = presource['pt_position_y'] * 4
    presource['pt_zt_μm'] = presource['pt_position_z'] * 40
    
    boxsource['pt_xt_μm'] = boxsource['pt_position_x'] * 4
    boxsource['pt_yt_μm'] = boxsource['pt_position_y'] * 4
    boxsource['pt_zt_μm'] = boxsource['pt_position_z'] * 40
    
    postsource['pt_xt_μm'] = postsource['pt_position_x'] * 4
    postsource['pt_yt_μm'] = postsource['pt_position_y'] * 4
    postsource['pt_zt_μm'] = postsource['pt_position_z'] * 40
    


def leveling_three(presource, boxsource, postsource):
    
    # pre
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', presource)
    X_transformed = np.array(X_transformed)
    presource['pt_xt_μm'] = X_transformed[:,0]
    presource['pt_yt_μm'] = X_transformed[:,1]
    presource['pt_zt_μm'] = X_transformed[:,2]  
    
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', boxsource)
    X_transformed = np.array(X_transformed)
    boxsource['pt_xt_μm'] = X_transformed[:,0]
    boxsource['pt_yt_μm'] = X_transformed[:,1]
    boxsource['pt_zt_μm'] = X_transformed[:,2]  
    
    #post
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', postsource)
    X_transformed = np.array(X_transformed)
    postsource['pt_xt_μm'] = X_transformed[:,0]
    postsource['pt_yt_μm'] = X_transformed[:,1]
    postsource['pt_zt_μm'] = X_transformed[:,2]



def leveling_four(source_one, source_two, source_three, source_four):
    
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', source_one)
    X_transformed = np.array(X_transformed)
    source_one['pt_xt_μm'] = X_transformed[:,0]
    source_one['pt_yt_μm'] = X_transformed[:,1]
    source_one['pt_zt_μm'] = X_transformed[:,2] 
    
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', source_two)
    X_transformed = np.array(X_transformed)
    source_two['pt_xt_μm'] = X_transformed[:,0]
    source_two['pt_yt_μm'] = X_transformed[:,1]
    source_two['pt_zt_μm'] = X_transformed[:,2] 
    
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', source_three)
    X_transformed = np.array(X_transformed)
    source_three['pt_xt_μm'] = X_transformed[:,0]
    source_three['pt_yt_μm'] = X_transformed[:,1]
    source_three['pt_zt_μm'] = X_transformed[:,2] 
    
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', source_four)
    X_transformed = np.array(X_transformed)
    source_four['pt_xt_μm'] = X_transformed[:,0]
    source_four['pt_yt_μm'] = X_transformed[:,1]
    source_four['pt_zt_μm'] = X_transformed[:,2] 


    
def conversion_four(source_one, source_two, source_three, source_four):
    
    source_one['pt_position_x_nm'] = source_one['pt_position_x'] * 4
    source_one['pt_position_y_nm'] = source_one['pt_position_y'] * 4
    source_one['pt_position_z_nm'] = source_one['pt_position_z'] * 40
    
    source_two['pt_position_x_nm'] = source_two['pt_position_x'] * 4
    source_two['pt_position_y_nm'] = source_two['pt_position_y'] * 4
    source_two['pt_position_z_nm'] = source_two['pt_position_z'] * 40
    
    source_three['pt_position_x_nm'] = source_three['pt_position_x'] * 4
    source_three['pt_position_y_nm'] = source_three['pt_position_y'] * 4
    source_three['pt_position_z_nm'] = source_three['pt_position_z'] * 40
    
    source_four['pt_position_x_nm'] = source_four['pt_position_x'] * 4
    source_four['pt_position_y_nm'] = source_four['pt_position_y'] * 4
    source_four['pt_position_z_nm'] = source_four['pt_position_z'] * 40


def conversion_five(presource_py, presource_bc, boxsource, postsource_py, postsource_bc):
    
    presource_py['pt_position_x_nm'] = presource_py['pt_position_x'] * 4
    presource_py['pt_position_y_nm'] = presource_py['pt_position_y'] * 4
    presource_py['pt_position_z_nm'] = presource_py['pt_position_z'] * 40
    
    presource_bc['pt_position_x_nm'] = presource_bc['pt_position_x'] * 4
    presource_bc['pt_position_y_nm'] = presource_bc['pt_position_y'] * 4
    presource_bc['pt_position_z_nm'] = presource_bc['pt_position_z'] * 40
    
    boxsource['pt_position_x_nm'] = boxsource['pt_position_x'] * 4
    boxsource['pt_position_y_nm'] = boxsource['pt_position_y'] * 4
    boxsource['pt_position_z_nm'] = boxsource['pt_position_z'] * 40
    
    postsource_py['pt_position_x_nm'] = postsource_py['pt_position_x'] * 4
    postsource_py['pt_position_y_nm'] = postsource_py['pt_position_y'] * 4
    postsource_py['pt_position_z_nm'] = postsource_py['pt_position_z'] * 40
    
    postsource_bc['pt_position_x_nm'] = postsource_bc['pt_position_x'] * 4
    postsource_bc['pt_position_y_nm'] = postsource_bc['pt_position_y'] * 4
    postsource_bc['pt_position_z_nm'] = postsource_bc['pt_position_z'] * 40



def leveling_five(presource_py, presource_bc, boxsource, postsource_py, postsource_bc):
    
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', presource_py)
    X_transformed = np.array(X_transformed)
    presource_py['pt_xt_μm'] = X_transformed[:,0]
    presource_py['pt_yt_μm'] = X_transformed[:,1]
    presource_py['pt_zt_μm'] = X_transformed[:,2] 
    
    # pre bc
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', presource_bc)
    X_transformed = np.array(X_transformed)
    presource_bc['pt_xt_μm'] = X_transformed[:,0]
    presource_bc['pt_yt_μm'] = X_transformed[:,1]
    presource_bc['pt_zt_μm'] = X_transformed[:,2] 
    
    # box
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', boxsource)
    X_transformed = np.array(X_transformed)
    boxsource['pt_xt_μm'] = X_transformed[:,0]
    boxsource['pt_yt_μm'] = X_transformed[:,1]
    boxsource['pt_zt_μm'] = X_transformed[:,2] 
    
    # post p
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', postsource_py)
    X_transformed = np.array(X_transformed)
    postsource_py['pt_xt_μm'] = X_transformed[:,0]
    postsource_py['pt_yt_μm'] = X_transformed[:,1]
    postsource_py['pt_zt_μm'] = X_transformed[:,2] 
    
    # post bc
    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', postsource_bc)
    X_transformed = np.array(X_transformed)
    postsource_bc['pt_xt_μm'] = X_transformed[:,0]
    postsource_bc['pt_yt_μm'] = X_transformed[:,1]
    postsource_bc['pt_zt_μm'] = X_transformed[:,2] 



def twod_graph(data, size, color, label, layer_23_to_4, layer_4_to_5, layer_5_to_6):
    fig, ax = plt.subplots(figsize = (5,4))
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = data.sample(size), ax = ax, color = color, label = label)
    ax.invert_yaxis()
    sns.despine(ax = ax)
    ax.legend()
    layer_lines(layer_23_to_4, layer_4_to_5, layer_5_to_6)
    



def twod_graph_pybcbox(presource_one, presource_two, boxsource, postsource_one, postsource_two, presize_one, presize_two, boxsize, postsize_one, postsize_two, label_pre_one, label_pre_two, label_box, label_post_one, label_post_two, layer_23_to_4, layer_4_to_5, layer_5_to_6):

    fig, ax = plt.subplots(figsize = (5, 4,))
    # pre 23p
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = presource_one.sample(presize_one), ax = ax, color = 'blue', label = label_pre_one)
    
    # pre bc
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = presource_two.sample(presize_two), ax = ax, color = 'red', label = label_pre_two)
    
    # box
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 30, data = boxsource.sample(boxsize), ax = ax, color = 'black', label = label_box)
    
    # post 23p
    sns.scatterplot( x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = postsource_one.sample(postsize_one), ax = ax, color = 'cyan', label = label_post_one)
    
    # post bc
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = postsource_two.sample(postsize_two), ax = ax, color = 'orange', label = label_post_two)

    ax.invert_yaxis()
    sns.despine(ax=ax)
    ax.legend()
    layer_lines(layer_23_to_4, layer_4_to_5, layer_5_to_6)
    


def twod_graph_four(source_one, source_two, source_three, source_four, sourcesize_one, sourcesize_two, sourcesize_three, sourcesize_four, label_one, label_two, label_three, label_four, layer_23_to_4, layer_4_to_5, layer_5_to_6):

    fig, ax = plt.subplots(figsize = (5, 4,))
    
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = source_one.sample(sourcesize_one), ax = ax, color = '#253494', label = label_one)
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = source_two.sample(sourcesize_two), ax = ax, color = '#2c7fc4', label = label_two)
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = source_three.sample(sourcesize_three), ax = ax, color = '#41b6c4', label = label_three)
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = source_four.sample(sourcesize_four), ax = ax, color = '#a1dab4', label = label_four)

    ax.invert_yaxis()
    
    ax.set_xlabel('x (μm)', fontsize=24)
    ax.set_ylabel('y (μm)', fontsize=24)
    
    ax.tick_params(axis='both', which='minor', labelsize=24)
    
    sns.despine(ax=ax)
    
    layer_lines(layer_23_to_4, layer_4_to_5, layer_5_to_6)



def twod_graph_inboxout(presource, boxsource, postsource, presize, boxsize, postsize, label_pre, label_box, label_post, layer_23_to_4, layer_4_to_5, layer_5_to_6):

    fig, ax = plt.subplots(figsize = (5, 4,))
    # pre
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = presource.sample(presize), ax = ax, color = 'red', label = label_pre)

    # box
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 30, data = boxsource.sample(boxsize), ax = ax, color = 'black', label = label_box)

    # post
    sns.scatterplot(x = 'pt_xt_μm', y = 'pt_yt_μm', s = 3, data = postsource.sample(postsize), ax = ax, color = 'orange', label = label_post)

    ax.invert_yaxis()
    sns.despine(ax=ax)
    ax.legend()
    layer_lines(layer_23_to_4, layer_4_to_5, layer_5_to_6)



def layer_lines(layer_23_to_4, layer_4_to_5, layer_5_to_6):
    plt.axhline(y = layer_23_to_4, color = 'black', linewidth=2)
    plt.axhline(y = layer_4_to_5, color = 'black', linewidth=2)
    plt.axhline(y = layer_5_to_6, color = 'black', linewidth=2)



def threed_graph_five(presource_one, presource_two, boxsource, postsource_one, postsource_two, presize_one, presize_two, boxsize, postsize_one, postsize_two, label_pre_one, label_pre_two, label_box, label_post_one, label_post_two, ax):
    
    pre_one_sample = presource_one.sample(presize_one)
    pre_two_sample = presource_two.sample(presize_two)
    box_sample = boxsource.sample(boxsize)
    post_one_sample = postsource_one.sample(postsize_one)
    post_two_sample = postsource_two.sample(postsize_two)
    
    # pre py
    ax.scatter(pre_one_sample['pt_xt_μm'], pre_one_sample['pt_zt_μm'], pre_one_sample['pt_yt_μm'], s = 3, color = 'blue', label = label_pre_one, alpha = 0.6)
    
    # pre bc
    ax.scatter(pre_two_sample['pt_xt_μm'], pre_two_sample['pt_zt_μm'], pre_two_sample['pt_yt_μm'], s = 3, color = 'red', label = label_pre_two, alpha = 0.6)
    
    # box
    ax.scatter(box_sample['pt_xt_μm'], box_sample['pt_zt_μm'], box_sample['pt_yt_μm'], s = 30, color='black', label = label_box, alpha = 1.0)
    
    # post py
    ax.scatter(post_one_sample['pt_xt_μm'], post_one_sample['pt_zt_μm'], post_one_sample['pt_yt_μm'], s = 3, color = 'cyan', label = label_post_one, alpha = 0.6)
    
    # post bc
    ax.scatter(post_two_sample['pt_xt_μm'], post_two_sample['pt_zt_μm'], post_two_sample['pt_yt_μm'], s = 3, color = 'orange', label = label_post_two, alpha = 0.6)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Z (μm)')
    ax.set_zlabel('Y (μm)')
    
    # Invert Z-axis (to have z (which is using y numbers and was relabeled to increase with depth)
    ax.invert_zaxis()
    
    # Add legend
    ax.legend()



def threed_graph_three(presource, boxsource, postsource, presize, boxsize, postsize, label_pre, label_box, label_post, ax):
    
    pre_cell_sample = presource.sample(presize)
    box_sample = boxsource.sample(boxsize)
    post_cell_sample = postsource.sample(postsize)
    
    # pre
    ax.scatter(pre_cell_sample['pt_xt_μm'], pre_cell_sample['pt_zt_μm'], pre_cell_sample['pt_yt_μm'], s=3, color='red', label=label_pre, alpha=0.6)
    
    # box
    ax.scatter(box_sample['pt_xt_μm'], box_sample['pt_zt_μm'], box_sample['pt_yt_μm'], s=30, color='black', label=label_box, alpha=1.0)
    
    # post
    ax.scatter(post_cell_sample['pt_xt_μm'], post_cell_sample['pt_zt_μm'],post_cell_sample['pt_yt_μm'], s=3, color='orange', label= label_post, alpha=0.6)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Z (μm)')
    ax.set_zlabel('Y (μm)')
    
    # Invert Z-axis (to have z (which is using y numbers and was relabeled to increase with depth)
    ax.invert_zaxis()
    
    # Add legend
    ax.legend()
    


def layer_lines(layer_23_to_4, layer_4_to_5, layer_5_to_6):
    
    plt.axhline(y = layer_23_to_4)
    plt.axhline(y = layer_4_to_5)
    plt.axhline(y = layer_5_to_6)


    
def analysis(source):
    cell_groups = {
        'Excitatory Neurons': ['23P', '4P', '5P-IT', '5P-ET', '5P-NP', '6P-IT', '6P-CT'],
        'Inhibitory Neurons': ['BC', 'MC', 'BPC', 'NGC'],
        'Glia Cells': ['astrocyte', 'oligo', 'OPC', 'microglia'],
        'Vascular Cells': ['pericyte']}
    
    # function for groups (cell type)
    def assign_group(cell_type):
        for group, types in cell_groups.items():
            if cell_type in types:
                return group
        return 'Other'
    
    # data frame
    result_df = pd.DataFrame({'count': source['cell_type'].value_counts(),'percent of all cells': source['cell_type'].value_counts(normalize = True) * 100}).sort_index()
    
    result_df['group'] = result_df.index.map(assign_group)
    
    # filtering neurons
    neurons = cell_groups['Excitatory Neurons'] + cell_groups['Inhibitory Neurons']
    all_cell_types = cell_groups['Excitatory Neurons'] + cell_groups['Inhibitory Neurons'] + cell_groups['Glia Cells'] + cell_groups['Vascular Cells']
    
    # creating a column 'only neurons' and add cell types
    result_df['only neurons'] = result_df.index.map(lambda x: assign_group(x) if x in neurons else 'X')
    result_df['all cells'] = result_df.index.map(lambda x: assign_group(x) if x in all_cell_types else 'X')
    
    # neuron percentages
    neuron_mask = source['cell_type'].isin(neurons)
    result_df['percent of neurons'] = (source[neuron_mask]['cell_type'].value_counts(normalize=True) * 100)
    
    # percentage by cell category
    # neurons are already labeled
    cell_mask = source['cell_type'].isin(all_cell_types)
    result_df['percent of cell types'] = (source[cell_mask]['cell_type'].value_counts(normalize=True) * 100)
    
    print(f"Total number of neurons (without Glia and Vascular Cells): {neuron_mask.sum()}")
    print(f"\nExcitatory: {source['cell_type'].isin(cell_groups['Excitatory Neurons']).sum()}")
    print(f"Inhibitory: {source['cell_type'].isin(cell_groups['Inhibitory Neurons']).sum()}")
    
    print(f"\nPercentage by CELL type:")
    print(result_df.groupby('all cells')['percent of cell types'].sum().sort_values(ascending = False))
    
    print(f"\nPercentage by NEURON type:")
    print(result_df.groupby('only neurons')['percent of neurons'].sum().sort_values(ascending = False))
    
    print(f"\nResults:")
    print(result_df)



def inputs(client, source):

    # Query synapse table with synapse_query()
    input_syn_df = client.materialize.synapse_query(post_ids = source)
    
    final_input_df = input_syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).count()[['id']].rename(columns = {'id': 'syn_count'}).sort_values(by = 'syn_count', ascending = False,).reset_index()
    pre_cells = final_input_df['pre_pt_root_id'].tolist()
    inputs_df = client.materialize.tables.aibs_metamodel_celltypes_v661(pt_root_id = pre_cells).query(split_positions = True, select_columns = ['pt_position', 'pt_root_id', 'cell_type'], limit = 1000000)
    return inputs_df, final_input_df



def outputs(client, source):

    # Query synapse table with synapse_query()
    output_syn_df = client.materialize.synapse_query(pre_ids = source)
    
    final_output_df = output_syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).count()[['id']].rename(columns = {'id': 'syn_count'}).sort_values(by = 'syn_count', ascending = False,).reset_index()
    post_cells = final_output_df['post_pt_root_id'].tolist()
    outputs_df = client.materialize.tables.aibs_metamodel_celltypes_v661(pt_root_id = post_cells).query(split_positions=True, select_columns = ['pt_position', 'pt_root_id', 'cell_type'], limit = 1000000)
    return outputs_df, final_output_df



def enlarging_box_with_cell_type(client, cell_type, number_of_cells, center_x, center_y, center_z):
    center = [center_x, center_y, center_z]
    expansion = 1

    while True:
        enlarging_bounding_box = [[center[0] - expansion, center[1] - expansion, center[2] - expansion], [center[0] + expansion, center[1] + expansion, center[2] + expansion]] # the coordinates are in voxels | center point coordinates [218809, 161360, 21251]
        enlarging_cell_type_bounding_box = client.materialize.tables.aibs_metamodel_celltypes_v661(cell_type = cell_type, pt_position_bbox = enlarging_bounding_box).query(split_positions = True)
        all_cells = enlarging_cell_type_bounding_box['pt_root_id'].tolist()
        cell_count = len(all_cells)
        if cell_count >= number_of_cells:
                break
        expansion += 100

    print('number of cells (type:', cell_type,') found:', cell_count)
    return all_cells

    
    
def movable_box_with_cell_type(client, cell_type, x_nm_min, y_nm_min, z_nm_min, x_nm_max, y_nm_max, z_nm_max):
    def nm_to_voxel_coordinates(x_nm_min, y_nm_min, z_nm_min, x_nm_max, y_nm_max, z_nm_max):

        transformed_coords = np.array([[x_nm_min, y_nm_min, z_nm_min], [x_nm_max, y_nm_max, z_nm_max]])
        
        voxel_coords = minnie_ds.transform_vx.invert(transformed_coords)
        
        x_voxel_min = int(np.round(voxel_coords[0, 0]))
        y_voxel_min = int(np.round(voxel_coords[0, 1]))
        z_voxel_min = int(np.round(voxel_coords[0, 2]))
        
        x_voxel_max = int(np.round(voxel_coords[1, 0]))
        y_voxel_max = int(np.round(voxel_coords[1, 1]))
        z_voxel_max = int(np.round(voxel_coords[1, 2]))
        
        return x_voxel_min, y_voxel_min, z_voxel_min, x_voxel_max, y_voxel_max, z_voxel_max
    
    x_voxel_min, y_voxel_min, z_voxel_min, x_voxel_max, y_voxel_max, z_voxel_max = \
        nm_to_voxel_coordinates(x_nm_min, y_nm_min, z_nm_min, x_nm_max, y_nm_max, z_nm_max)
    
    bounding_box = [[x_voxel_min, y_voxel_min, z_voxel_min], [x_voxel_max, y_voxel_max, z_voxel_max]]
    
    cell_df = client.materialize.tables.aibs_metamodel_celltypes_v661(cell_type = cell_type, pt_position_bbox = bounding_box).query(split_positions = True)
    
    all_cells = cell_df['pt_root_id'].tolist()
    
    print(f"Found {len(all_cells)} cells of type '{cell_type}'")
    return all_cells, cell_df



def voxel_to_nm_coordinates(x_vox_min, y_vox_min, z_vox_min, x_vox_max, y_vox_max, z_vox_max):
    voxel_coords = np.array([[x_vox_min, y_vox_min, z_vox_min], [x_vox_max, y_vox_max, z_vox_max]])

    nm_coords = minnie_ds.transform_vx.apply(voxel_coords)
    
    x_nm_min = nm_coords[0, 0]
    y_nm_min = nm_coords[0, 1]
    z_nm_min = nm_coords[0, 2]
    
    x_nm_max = nm_coords[1, 0]
    y_nm_max = nm_coords[1, 1]
    z_nm_max = nm_coords[1, 2]
    
    return x_nm_min, y_nm_min, z_nm_min, x_nm_max, y_nm_max, z_nm_max



def um_to_voxels(x_um, y_um, z_um):

    # array
    transformed_coords = np.column_stack((x_um, y_um, z_um))
    
    # inverse transformation
    original_voxels = minnie_ds.transform_vx.invert(transformed_coords)
    
    # round
    original_voxels_int = np.round(original_voxels).astype(int)
    
    # individual results
    x_vox = original_voxels_int[:, 0]
    y_vox = original_voxels_int[:, 1]
    z_vox = original_voxels_int[:, 2]
    
    print(f"voxels: x={x_vox}, y={y_vox}, z={z_vox}")
    
    return x_vox, y_vox, z_vox



def voxels_to_um(x_vox, y_vox, z_vox):
    
    # array
    voxel_coords = np.column_stack((x_vox, y_vox, z_vox))

    # transformation
    transformed_nm = minnie_ds.transform_vx.apply(voxel_coords)

    # individual results
    x_um = transformed_nm[:, 0]
    y_um = transformed_nm[:, 1]
    z_um = transformed_nm[:, 2]
    
    print(f"microns: x={x_um}, y={y_um}, z={z_um}")
    
    return x_um, y_um, z_um



def bounding_box(x_min_um, y_min_um, z_min_um, x_max_um, y_max_um, z_max_um):
    
    x_min_vox, y_min_vox, z_min_vox = um_to_voxels(x_min_um, y_min_um, z_min_um)
    x_max_vox, y_max_vox, z_max_vox = um_to_voxels(x_max_um, y_max_um, z_max_um)
    
    coordinates_minimum_voxels_int = [int(x_min_vox[0]), int(y_min_vox[0]), int(z_min_vox[0])]
    coordinates_maximum_voxels_int = [int(x_max_vox[0]), int(y_max_vox[0]), int(z_max_vox[0])]


    bounding_box_return = [coordinates_minimum_voxels_int, coordinates_maximum_voxels_int]
    return bounding_box_return



def coordinates_from_root_id(root_id_list):
    my_root_ids = root_id_list
    
    coordinates_df = client.materialize.tables.aibs_metamodel_celltypes_v661(pt_root_id = my_root_ids).query(split_positions = False)
    coordinates = coordinates_df[['pt_root_id', 'cell_type', 'pt_position']]

    coordinates_split_df = client.materialize.tables.aibs_metamodel_celltypes_v661(pt_root_id = my_root_ids).query(split_positions = True)
    coordinates_split = coordinates_split_df[['pt_root_id', 'cell_type', 'pt_position_x', 'pt_position_y', 'pt_position_z']]
    
    coordinate_x_vox = coordinates_split_df['pt_position_x'].values[0]
    coordinate_y_vox = coordinates_split_df['pt_position_y'].values[0]
    coordinate_z_vox = coordinates_split_df['pt_position_z'].values[0]

    return coordinates, coordinates_split, coordinate_x_vox, coordinate_y_vox, coordinate_z_vox



def find_cells_in_radius(client, center_cell_id, coordinate_x_vox, coordinate_y_vox, coordinate_z_vox, min_radius_um, max_radius_um, cell_df):

    x_um, y_um, z_um = voxels_to_um(coordinate_x_vox, coordinate_y_vox, coordinate_z_vox)
    center_coords = np.array([x_um, y_um, z_um]).flatten()

    X_transformed = minnie_ds.transform_vx.apply_dataframe('pt_position', cell_df)
    X_transformed = np.array(X_transformed)
    
    cell_df_copy = cell_df.copy()
    cell_df_copy['pt_xt_μm'] = X_transformed[:, 0]
    cell_df_copy['pt_yt_μm'] = X_transformed[:, 1]
    cell_df_copy['pt_zt_μm'] = X_transformed[:, 2]

    # distance
    all_coords = cell_df_copy[['pt_xt_μm', 'pt_yt_μm', 'pt_zt_μm']].values
    distances = np.linalg.norm(all_coords - center_coords, axis = 1)
    cell_df_copy['distance_um'] = distances
    
    # get cells within radius
    cells_in_radius_copy = cell_df_copy[(cell_df_copy['distance_um'] >= min_radius_um) & (cell_df_copy['distance_um'] <= max_radius_um)].copy()
    cells_in_radius = cells_in_radius_copy.sort_values('distance_um')
    
    print(f"found: {len(cells_in_radius)} cells within radius of {min_radius_um} μm to {max_radius_um} μm")
    print(f"central cell: {center_cell_id}")
    
    return cells_in_radius, center_coords



def distance_dependency_bc_to_23p(cells_within_radius_df, center_cell_id):
    # Cells in radius
    cells_within_radius_list = cells_within_radius_df['valid_id'].tolist()
    print('cells_within_radius_list', cells_within_radius_list)
    number_cell_type_two = len(cells_within_radius_list)
    number_cell_type_one = len(center_cell_id)
    print('number_cell_type_two', number_cell_type_two)
    print('number_cell_type_one', number_cell_type_one)
    
    output_syn_df = client.materialize.synapse_query(pre_ids = center_cell_id, post_ids = cells_within_radius_list)
    output_df = output_syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).count()[['id']].rename(columns = {'id': 'syn_count'}).sort_values(by='syn_count', ascending = False).reset_index()
    
    unique_connections_pre_post_df = output_df[['pre_pt_root_id', 'post_pt_root_id']].drop_duplicates()
    number_connections_found_pre_post = len(unique_connections_pre_post_df)
    print('number_connections_found BC->23P (unique pairs):', number_connections_found_pre_post)

    input_syn_df = client.materialize.synapse_query(pre_ids = cells_within_radius_list, post_ids = center_cell_id)
    input_df = input_syn_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).count()[['id']].rename(columns = {'id': 'syn_count'}).sort_values(by='syn_count', ascending = False).reset_index()
    
    unique_connections_post_pre_df = input_df[['pre_pt_root_id', 'post_pt_root_id']].drop_duplicates()
    number_connections_found_post_pre = len(unique_connections_post_pre_df)
    print('number_connections_found 23P->BC (unique pairs):', number_connections_found_post_pre)
    
    reciprocal_df = pd.merge(output_df, input_df, left_on = ['pre_pt_root_id', 'post_pt_root_id'], right_on = ['post_pt_root_id', 'pre_pt_root_id'], suffixes = ['_BC_to_23P', '_23P_to_BC'])
    number_reciprocal_found = len(reciprocal_df)
    print(f'\nReciprocal connections found: {number_reciprocal_found}')

    number_cells_total = number_cell_type_one + number_cell_type_two
    print(f'\n--- Population sizes ---')
    print(f'BC cells: {number_cell_type_one}')
    print(f'23P cells in radius: {number_cell_type_two}')
    print(f'Total cells: {number_cells_total}')
    
    total_possible_connections = (number_cell_type_one) * (number_cell_type_two)
    print(f'\n--- Connectivity ---')
    print(f'Total possible connections (BC->23P or 23P->BC): {total_possible_connections}')
    
    connectivity_pre_post = (number_connections_found_pre_post / total_possible_connections)
    connectivity_pre_post_percent = connectivity_pre_post * 100
    print(f'Connectivity BC->23P: {connectivity_pre_post_percent:.4f}%')
    
    connectivity_post_pre = (number_connections_found_post_pre / total_possible_connections)
    connectivity_post_pre_percent = connectivity_post_pre * 100
    print(f'Connectivity 23P->BC: {connectivity_post_pre_percent:.4f}%')

    print(f'\n--- Reciprocity ---')
    
    reciprocity_prob_chance = connectivity_pre_post * connectivity_post_pre
    print(f'Probability of reciprocity by chance: {reciprocity_prob_chance * 100:.4f}%')
    
    expected_reciprocal_count = reciprocity_prob_chance * total_possible_connections
    print(f'Expected reciprocal pairs (by chance): {expected_reciprocal_count:.2f}')

    if total_possible_connections > 0:
        reciprocity_rate_observed_percent = (number_reciprocal_found / total_possible_connections) * 100
    else:
        reciprocity_rate_observed_percent = 0
    
    print(f'Reciprocity observed (population wide): {reciprocity_rate_observed_percent:.4f}%')
    print(f'Count: {number_reciprocal_found} observed vs. {expected_reciprocal_count:.2f} expected')

    if number_reciprocal_found > expected_reciprocal_count:
        print('RESULT: Reciprocity is OVERREPRESENTED')
    elif number_reciprocal_found < expected_reciprocal_count:
        print('RESULT: Reciprocity is UNDERREPRESENTED')
    else:
        print('RESULT: Reciprocity matches chance expectation')

    if number_reciprocal_found > 0:
        print('\nReciprocal pairs:')
        print(reciprocal_df[['pre_pt_root_id_BC_to_23P', 'post_pt_root_id_BC_to_23P', 'syn_count_BC_to_23P', 'syn_count_23P_to_BC']])



def get_center_coords_um(root_id):
    coords, _, x_vox, y_vox, z_vox = coordinates_from_root_id(root_id)
    x_um_arr, y_um_arr, z_um_arr = voxels_to_um([x_vox], [y_vox], [z_vox])
    return np.array([x_um_arr[0], y_um_arr[0], z_um_arr[0]]), x_vox, y_vox, z_vox