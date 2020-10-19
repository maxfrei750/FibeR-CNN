import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def display_image(image, **kwargs):
    plt.imshow(image, **kwargs)
    plt.axis("off")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show(block=True)


def get_viridis_colors(num_colors):
    color_min = (0.231, 0.322, 0.545)
    color_max = (0.369, 0.788, 0.384)

    if num_colors == 1:
        return color_min

    colormap = LinearSegmentedColormap.from_list("custom_viridis", colors=[color_min, color_max])

    colors = []

    for i in range(num_colors):
        color = colormap(i / (num_colors - 1))[:3]
        colors.append(color)

    return colors
