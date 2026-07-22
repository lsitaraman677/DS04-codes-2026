import cv2
import sys
import os

def main():
    # Check if a file path was passed via the command line
    if len(sys.argv) < 2:
        print("Usage: python view.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Verify the file actually exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not decode image '{image_path}'.")
        sys.exit(1)

    # Set up the window title
    window_name = os.path.basename(image_path)

    # WINDOW_NORMAL allows the window to be resized by the user
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)

    print(f"Viewing: {image_path}")
    print("Press any key or click the 'X' to close.")

    # Loop to keep the window open until a key is pressed or the window is closed
    while True:
        # Check if the user clicked the 'X' button
        # WND_PROP_VISIBLE returns 0 if the window has been closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Wait 50ms for a key press; breaks the loop if any key is pressed
        if cv2.waitKey(50) != -1:
            break

    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
