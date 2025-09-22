from PIL import Image
import os


def find_most_similar_orientation(player_image, sprites_dir="sprites", alpha_threshold=10):

    directions = ["up", "down", "left", "right"]
    best_similarity = float('inf')
    best_direction = None

    player_rgb = player_image.convert('RGB')
    player_pixels = list(player_rgb.getdata())

    for direction in directions:
        sprite_path = os.path.join(sprites_dir, f"{direction}.png")

        if not os.path.exists(sprite_path):
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = os.path.join(sprites_dir, direction, f"{direction}{ext}")
                if os.path.exists(test_path):
                    sprite_path = test_path
                    break
            else:
                continue

        try:
            sprite = Image.open(sprite_path)
            sprite = sprite.resize(player_image.size)

            if sprite.mode != 'RGBA':
                sprite = sprite.convert('RGBA')

            sprite_pixels = list(sprite.getdata())

            total_pixels = 0
            mse_sum = 0

            for i, (p1, p2) in enumerate(zip(player_pixels, sprite_pixels)):
                # Skip transparent pixels based on threshold
                if p2[3] < alpha_threshold:
                    continue

                r1, g1, b1 = p1
                r2, g2, b2 = p2[:3]

                mse_sum += (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2
                total_pixels += 1

            if total_pixels == 0:
                mse = float('inf')
            else:
                mse = mse_sum / total_pixels

            if mse < best_similarity:
                best_similarity = mse
                best_direction = direction

            print(direction, mse)

        except Exception as e:
            print(f"Error processing {sprite_path}: {e}")

    return best_direction