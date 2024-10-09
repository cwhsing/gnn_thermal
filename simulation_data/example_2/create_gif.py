from PIL import Image


def create_gif(image_paths, output_gif_path, duration=20):
    images = [Image.open(image_path) for image_path in image_paths]
    # Save as GIF
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


if __name__ == "__main__":
    # List of image file paths
    image_paths = [f'figure_{step:03d}.png' for step in range(801)]  # Add your file paths
    # Output GIF path
    output_gif_path = "data_19.gif"
    # Create GIF
    create_gif(image_paths, output_gif_path)
