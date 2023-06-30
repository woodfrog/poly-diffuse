import numpy as np
import cv2


def process_floorplan(data):
    corners = set()
    edges = list()
    for room in data:
        for vert in room:
            corners.add(tuple(vert))
    corners = list(corners)
    corner_to_id = {c:idx for idx, c in enumerate(corners)}
    for room in data:
        for idx in range(len(room)):
            c1 = corner_to_id[tuple(room[idx])]
            idx_next = (idx + 1) % len(room)
            c2 = corner_to_id[tuple(room[idx_next])]
            edges.append((c1, c2))
    corners = np.array(corners).astype(np.float32)
    regions = [np.array(item).astype(np.float32) for item in data]
    return corners, edges, regions


colors_12 = [
    "#DCECC9",
    "#B3DDCC",
    "#8ACDCE",
    "#62BED2",
    "#46AACE",
    "#3D91BE",
    "#3677AE",
    "#2D5E9E",
    "#24448E",
    "#1C2B7F",
    "#162165",
    "#11174B",
]

colors_new = [
    '#3a86ff',
    '#F94144',
    '#F8961E',
    '#ffd670',
    '#90BE6D',
    '#4D908E',
    '#577590',
    '#277DA1',
    '#8338ec',
]


def plot_floorplan_with_regions(regions, corners, edges, scale, sort_regions=True):
    colors = colors_12[:8]
    #colors = colors_new

    regions = [(region * scale / 256).round().astype(np.int) for region in regions]
    corners = (corners * scale / 256).round().astype(np.int)

    # define the color map
    room_colors = [colors[i % 8] for i in range(len(regions))]
    #room_colors = [colors[i] for i in range(len(regions))]

    colorMap = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in room_colors]
    colorMap = np.asarray(colorMap)
    if len(regions) > 0:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap], axis=0).astype(
            np.uint8)
    else:
        colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0)], axis=0).astype(
            np.uint8)
    # when using opencv, we need to flip, from RGB to BGR
    colorMap = colorMap[:, ::-1]

    alpha_channels = np.zeros(colorMap.shape[0], dtype=np.uint8)
    alpha_channels[1:len(regions) + 1] = 150

    colorMap = np.concatenate([colorMap, np.expand_dims(alpha_channels, axis=-1)], axis=-1)

    room_map = np.zeros([scale, scale]).astype(np.int32)
    # sort regions
    if sort_regions and len(regions) > 1:
        avg_corner = [region.mean(axis=0) for region in regions]
        ind = np.argsort(np.array(avg_corner)[:, 0], axis=0)
        regions = np.array(regions)[ind]

    for idx, polygon in enumerate(regions):
        cv2.fillPoly(room_map, [polygon], color=idx + 1)

    image = colorMap[room_map.reshape(-1)].reshape((scale, scale, 4))

    pointColor = tuple((np.array([0.95, 0.3, 0.3, 1]) * 200).astype(np.uint8).tolist())
    for point in corners:
        cv2.circle(image, tuple(point), color=pointColor, radius=12, thickness=-1)
        cv2.circle(image, tuple(point), color=(255, 255, 255, 255), radius=6, thickness=-1)

    for edge in edges:
        c1 = corners[edge[0]]
        c2 = corners[edge[1]]
        cv2.line(image, tuple(c1), tuple(c2), color=(0, 0, 0, 255), thickness=3)

    return image
