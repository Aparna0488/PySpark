from spark_init import sc


def euclidean_distance(a, b):
    return sum((c1 - c2) ** 2 for c1, c2 in zip(a, b)) ** 0.5

def closest_centroid(point, centroids):
    min_dist = float("inf")
    min_centroid = 0
    for i in range(len(centroids)):
        distance = euclidean_distance(point, centroids[i])
        if min_dist > distance:
            min_dist = distance
            min_centroid = i+1
        # print(distance)
    return ("C"+str(min_centroid), point)

def agg(x,y):
    sum = []

    for i in range(len(x)):
        sum.append(x[i] + y[i])
    return sum

def avg(x, count):
    avg = []
    for i in range(len(x[1])):
        avg.append(x[1][i]/count[x[0]])
    return avg

K = 2
maxIter = 10

points = sc.textFile('kmeans_data.txt').map(lambda x: [float(num) for num in x.split(' ')])
centroids = sc.broadcast(points.takeSample(withReplacement=False, num=K, seed=3))
centroids_value = centroids.value

for _ in range(maxIter):
    # MapReduce-based KMeans method to calculate and update centroids
    res_map = points.map(lambda point: closest_centroid(point, centroids_value))
    # print(res_map.collect())
    res_map_count = res_map.countByKey()
    # print(res_map_count)

    res_reduce = res_map.reduceByKey(lambda x, y: agg(x, y)).map(lambda x: avg(x, res_map_count))
    # print(res_reduce.collect())
    centroids_value = res_reduce.collect()
print(centroids_value)