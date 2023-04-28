import csv
import pandas as pd
import numpy as np
import re
import json
from numpy.linalg import norm

def prepare_for_weka(old_df, new_df, unknown="Unknown"):
    df = pd.read_csv(old_df, encoding='latin')
    
    #Remove ','
    df = df.replace({"," : " "}, regex=True)
    
    #Remove special chars
    df = df.replace({"[^a-zA-Z0-9]" : " "}, regex=True)

    #Fix unknown values
    df = df.replace({unknown : "?"}, regex=True)

    df.to_csv(new_df, index=False)
    pass

def knn_rs():
    #create_merged_table()
    df_rtng = pd.read_csv("rating.csv", encoding="latin")
    #df_anime = pd.read_csv("real.csv", encoding="latin")

    n_samples=7

    sample_df = df_rtng.groupby("user_id", group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples), random_state=123))
    print(sample_df)
    sample_df.to_csv("sampled_ratings.csv")
    df_rs = sample_df.pivot_table(index="anime_id", columns="user_id", values="rating").fillna(0)
    df_rs.to_csv("ratings_for_rs.csv")
    #mat_rs = csr_matrix(df_rs.values)
    #print(mat_rs)
    #sparse.save_npz("matrix.npz", mat_rs)

def main():
    #prepare_for_weka(old_df="anime.csv", new_df="real.csv")
    with open("sampled_ratings.csv", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [row for row in reader]

    users = set(row['user_id'] for row in data)

    map_dicts = {}
    with open("dict.json", "a") as f:
        for u in users:
            map_dicts[u] = {}
            for row in data:
                if row['user_id'] == u:
                    anime = row["anime_id"]
                    rating = row["rating"]
                    map_dicts[u][anime] = rating
            json_obj = json.dumps(map_dicts[u])
            f.write(json_obj + "\n")

    print(map_dicts)
    pass

def filter_rate(current_user, skip=-1):

    current_user = {key: int(value) for key, value in current_user.items()}
    values = [int(value) for value in current_user.values() if int(value) != skip]
    if(len(values) != 0): 
        avg = sum(values) / len(values)
        current_user = {key:avg if value == skip else value for key, value in current_user.items()}
    else:
        current_user = {}

    return current_user

def rs_user_to_user(user, item, kn=20):
    
    pairs = []

    file = open("dict.json")
    row = file.readlines()

    current_user = json.loads(row[user])

    current_user = filter_rate(current_user)
    values = [int(value) for value in current_user.values()]
    current_user_avg = sum(values) / len(values)

    #print("[+]Profile of current user: ", current_user)
    #print("\n")

    for line in row:
        my_dict = json.loads(line)

        if (str(item) in my_dict) == False:
            continue
        
        my_dict = filter_rate(my_dict)
        
        #print("[+]Selected user:", my_dict)

        keys = set(my_dict.keys()) & set(current_user.keys())
        v1 = []
        v2 = []

        for k in keys:
            v1.append(my_dict[k])
            v2.append(current_user[k])
        
        if(len(keys) == 0):
            #print("[-]No common items has been found!")
            pass
        else:
            #print("[+]Following items has been rated by both: ", keys)
            v1 = np.array(v1, dtype=float)
            v2 = np.array(v2, dtype=float)
            v1[v1 == -1] = np.mean(v1)
            v2[v2 == -1] = np.mean(v2)
            v1 = [x - np.mean(v1) for x in v1]
            v2 = [x - np.mean(v2) for x in v2]
            #print("[+]Calculating similarity among:", v1, "and", v2)

            #This user is represented by always same rate
            if(norm(v1)) == 0:
                #print("[!!]This user gives always same rate for these items. We cannot trace a good profile.\n")
                continue

            #Current user is represented by always same rate
            if(norm(v2)) == 0:
                #print("[-]Current user gives always same rate for these items. We cannot trace a good profile.")
                continue

            similarity = np.dot(v1, v2) / (norm(v1) * norm(v2))
            #print("[+]Similarity among:", v1, "and", v2)

            if(len(pairs) < kn):
                pairs.append((similarity, my_dict[item]))
            elif (len(pairs) == kn):
                 my_min = min(x[0] for x in pairs)
                 if similarity >= my_min:
                    x = 0
                    for i, p in enumerate(pairs):
                        if(p[0] == my_min):
                            x = i
                            break
                    pairs[x] = (similarity, my_dict[item])

            #print("[+]Similarity among", v1, "and", v2, "is", similarity)
        #print("\n")
    file.close()
    
    sim_vct = [x[0] for x in pairs]
    rate_vct = [x[1] for x in pairs]
    sum_sim_vct = sum(np.absolute(sim_vct))

    if sum_sim_vct == 0:
        return -1

    return current_user_avg + (np.dot(sim_vct, rate_vct)) / (sum(np.absolute(sim_vct)))

#For an item yet seen
print("score(0, p)    =", rs_user_to_user(0, "p"))
print("score(0, 4975) =", rs_user_to_user(0, "4975"))

#For new items
print("score(0, item) =", rs_user_to_user(0, "item"))
print("score(0, 376) =", rs_user_to_user(0, "376"))