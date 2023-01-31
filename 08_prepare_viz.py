import numpy as np
import umap

emb = np.load("<your_output_npy_file>")  # Generated .npy file in previous step
point_file = "<your_point_file>"  # E.g., ./points.txt
n_components = 2

mapper = umap.UMAP(n_neighbors=8,
                   min_dist=0.5,
                   n_components=n_components,
                   metric='euclidean')

# emb = np.load("./test.npy")  # Generated .npy file in previous step
# point_file = "./points.txt"  # E.g., ./points.txt
# n_components = 3

def save_2d_points(points, file_name):
    with open(file_name, 'w') as f:
        for i in range(len(points)):
            f.write(f"{points[i][0]},{points[i][1]}")
            if i != len(points) - 1:
                f.write("\n")


def save_3d_points(points, file_name):
    with open(file_name, 'w') as f:
        for i in range(len(points)):
            f.write(f"{points[i][0]},{points[i][1]},{points[i][2]}")
            if i != len(points) - 1:
                f.write("\n")


embedding = mapper.fit(emb)
if n_components == 3:
    save_3d_points(embedding.embedding_, point_file)
elif n_components == 2:
    save_2d_points(embedding.embedding_, point_file)

# video_path = "<your_video_root>"
# target_path = "<your_resource_root>"# df = pd.read_csv('<your_annotation_rooot>/rts_emo_test.csv')
# for i in df["video_id"]:
#     shutil.copy(f"{video_path}/{i.split('_')[0]}/{i}.mp4", target_path)
