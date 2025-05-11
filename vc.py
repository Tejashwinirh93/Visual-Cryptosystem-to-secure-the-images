import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk


# PSNR Calculation
def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100  # No error
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


# Normalized Cross-Correlation Calculation
def calculate_ncorr(original, reconstructed):
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    correlation = np.corrcoef(original_flat, reconstructed_flat)[0, 1]
    return correlation


# Logistic Map Function for Chaotic Sequence
def logistic_map(size, seed=0.5, r=3.9):
    sequence = np.zeros(size, dtype=np.float32)
    x = seed
    print(f"Logistic map input (Seed value): {seed}")
    for i in range(size):
        x = r * x * (1 - x)  # Logistic map equation
        sequence[i] = x
        if i < 1:  # Display first 10 outputs for brevity
            print(f"Logistic map output {i}: {x}")

    scaled_sequence = (sequence * 255).astype(np.uint8)
    return scaled_sequence

# Binarize the Image
def binarize_image(image, threshold=128):
    if len(image.shape) == 3:  # Colored image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  # Already grayscale
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


# Generate a Single Share Using XOR and Chaotic Sequence
def generate_single_share(binary_image, chaotic_sequence):
    height, width = binary_image.shape
    share1 = np.zeros((height, width), dtype=np.uint8)
    share2 = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            chaos_value = chaotic_sequence[i * width + j]  # Use chaotic sequence value
            if binary_image[i, j] == 255:  # White pixel
                share1[i, j] = chaos_value
                share2[i, j] = chaos_value  # Both shares are the same for white pixels
            else:  # Black pixel
                share1[i, j] = chaos_value
                share2[i, j] = 255 - chaos_value  # Shares are complementary
        else:
                chaos_binary = 255 if chaos_value > 128 else 0  # chaotic value is binarized into 0 or 255 based on a threshold of 128.
                # XOR with the binary image
                share1[i, j] = chaos_binary
                share2[i, j] = chaos_binary ^ binary_image[i, j]

    return share1, share2  # Return both share1 and share2


# Reconstruct the Image by XORing the Shares and darkening the result
def reconstruct_image_xor(share1, share2, darkening_factor=0.5):
    # Perform the XOR operation
    reconstructed_image = cv2.bitwise_xor(share1, share2)
    
    # Apply darkening by scaling pixel values
    darkened_image = np.uint8(reconstructed_image * darkening_factor)  # Multiply pixel values by darkening factor
    return darkened_image
   

# Visual Cryptography Processing for Colored Image
def process_visual_cryptography(image_path, output_folder, threshold=128, seed=0.7, darkening_factor=0.5):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    binary_image = binarize_image(image, threshold)

    height, width = binary_image.shape
    total_pixels = height * width
    chaotic_sequence = logistic_map(total_pixels, seed=seed)

    # Split the color channels (BGR)
    channels = cv2.split(image)

    share1_channels = []
    share2_channels = []

    # Generate shares for each color channel (R, G, B)
    for channel in channels:
        share1, share2 = generate_single_share(channel, chaotic_sequence)
        share1_channels.append(share1)
        share2_channels.append(share2)

    # Merge the shares for each channel back into the final share images
    share1 = cv2.merge(share1_channels)
    share2 = cv2.merge(share2_channels)

    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, 'share1.png'), share1)
    cv2.imwrite(os.path.join(output_folder, 'share2.png'), share2)

    # Reconstruct the image with darkening factor applied
    reconstructed_image = cv2.merge([reconstruct_image_xor(share1_channels[i], share2_channels[i], darkening_factor) for i in range(3)])
    cv2.imwrite(os.path.join(output_folder, 'reconstructed_image.png'), reconstructed_image)

    # Correctly call PSNR and NCORR with the actual reconstructed image
    psnr_xor = calculate_psnr(image, reconstructed_image)
    ncor_xor = calculate_ncorr(image, reconstructed_image)

    print(f"PSNR (XOR): {psnr_xor} dB")
    print(f"NCORR (XOR): {ncor_xor}")

    return share1, share2, reconstructed_image, psnr_xor, ncor_xor



# Generate a Single Share from Reconstructed Image
def generate_single_share_from_reconstructed(reconstructed_image, chaotic_sequence):
    binary_reconstructed_image = binarize_image(reconstructed_image)  # Convert to binary
    share1, share2 = generate_single_share(binary_reconstructed_image, chaotic_sequence)
    # Return only share1 (or both share1, share2 if needed)
    return share1


# Generate share based on reconstructed image
def generate_share_from_reconstructed(self):
    if not self.output_folder:
        messagebox.showerror("Error", "Please process an image first.")
        return

    reconstructed_image_path = os.path.join(self.output_folder, 'reconstructed_image.png')
    reconstructed_image = cv2.imread(reconstructed_image_path)

    if reconstructed_image is None:
        messagebox.showerror("Error", "Failed to load the reconstructed image.")
        return

    total_pixels = reconstructed_image.shape[0] * reconstructed_image.shape[1]
    chaotic_sequence = logistic_map(total_pixels, seed=0.7)
    new_share = generate_single_share_from_reconstructed(reconstructed_image, chaotic_sequence)

    if isinstance(new_share, np.ndarray):
        if new_share.shape[0] > 0 and new_share.shape[1] > 0:
            new_share_path = os.path.join(self.output_folder, 'new_share.png')
            cv2.imwrite(new_share_path, new_share)
            messagebox.showinfo("Success", "New share generated based on reconstructed image.")
            self.status_label.config(text="Status: New Share Generated", fg="#00ff00")
        else:
            messagebox.showerror("Error", "Generated share is empty.")
    else:
        messagebox.showerror("Error", "Generated share is not a valid NumPy array.")

# Decrypt the Image
def decrypt_image(share1_path, share2_path, output_path):
    share1 = cv2.imread(share1_path, cv2.IMREAD_COLOR)
    share2 = cv2.imread(share2_path, cv2.IMREAD_COLOR)

    # Split the shares into color channels
    share1_channels = cv2.split(share1)
    share2_channels = cv2.split(share2)

    # Reconstruct each color channel
    reconstructed_channels = [
        reconstruct_image_xor(share1_channels[i], share2_channels[i]) for i in range(3)
    ]

    # Merge channels to form the final image
    reconstructed_image = cv2.merge(reconstructed_channels)

    cv2.imwrite(output_path, reconstructed_image)
    print(f"Decrypted image saved as '{output_path}'.")
    

    
def reverse_xor_decryption(share1, share2):
    """This function performs the XOR operation to reconstruct the image."""
    return cv2.bitwise_xor(share1, share2)

class CreativeVisualCryptographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Creative Visual Cryptography Tool")
        self.root.geometry("1200x800")  # Increased size to fit all images
        self.root.configure(bg="#2a2a2a")

        self.image_path = None
        self.output_folder = None

        # Image display labels
        self.labels = {}

        # Title Label
        title_label = tk.Label(
            self.root,
            text="Visual Cryptography Tool",
            font=("Helvetica", 24, "bold"),
            bg="#2a2a2a",
            fg="#ffffff"
        )
        title_label.pack(pady=20)

        # Frame for Buttons
        frame = tk.Frame(self.root, bg="#2a2a2a")
        frame.pack(pady=20)

        # Buttons
        select_image_button = ttk.Button(frame, text="Select Image", command=self.select_image, style="Custom.TButton")
        select_image_button.grid(row=0, column=0, padx=10, pady=10)

        process_button = ttk.Button(frame, text="Process Image", command=self.process_image, style="Custom.TButton")
        process_button.grid(row=0, column=1, padx=10, pady=10)

        decrypt_button = ttk.Button(frame, text="Decrypt Image", command=self.decrypt_image, style="Custom.TButton")
        decrypt_button.grid(row=0, column=2, padx=10, pady=10)

        # Status Labels
        self.psnr_label = tk.Label(self.root, text="PSNR: N/A", font=("Helvetica", 14), bg="#2a2a2a", fg="#ffffff")
        self.psnr_label.pack(pady=5)

        self.ncorr_label = tk.Label(self.root, text="NCORR: N/A", font=("Helvetica", 14), bg="#2a2a2a", fg="#ffffff")
        self.ncorr_label.pack(pady=5)

        self.status_label = tk.Label(self.root, text="Status: Waiting for Action", font=("Helvetica", 14), bg="#2a2a2a", fg="#ffcc00")
        self.status_label.pack(pady=5)

        # Frame for Image Display
        self.image_display_frame = tk.Frame(self.root, bg="#2a2a2a")
        self.image_display_frame.pack(pady=10)

        # Custom Styles for Buttons
        style = ttk.Style()
        style.configure("Custom.TButton", font=("Helvetica", 12), width=20, relief="solid", borderwidth=2)

    def display_image(self, image_path, image_name):
        """Displays an image in the GUI with a label."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        if image_name in self.labels:
            self.labels[image_name].config(image=img)
            self.labels[image_name].image = img
        else:
            label = tk.Label(self.image_display_frame, image=img, bg="#2a2a2a")
            label.image = img
            label.pack(side=tk.LEFT, padx=10)
            self.labels[image_name] = label

    def select_image(self):
     self.image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    
     if not self.image_path:
        messagebox.showerror("Error", "No image selected.")
        return

     if not os.path.exists(self.image_path):
        messagebox.showerror("Error", "Selected file does not exist.")
        return
    
     self.status_label.config(text="Status: Image Selected", fg="#00ff00")
     self.display_image(self.image_path, "original")


    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        self.output_folder = filedialog.askdirectory(title="Select Output Folder")
        if not self.output_folder:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        share1, share2, reconstructed_image, psnr_xor, ncor_xor = process_visual_cryptography(
            self.image_path, self.output_folder
        )

        self.psnr_label.config(text=f"PSNR: {psnr_xor:.2f} dB")
        self.ncorr_label.config(text=f"NCORR: {ncor_xor:.2f}")
        self.status_label.config(text="Status: Image Processed", fg="#00ff00")

        # Display shares and reconstructed image
        self.display_image(os.path.join(self.output_folder, 'share1.png'), "share1")
        self.display_image(os.path.join(self.output_folder, 'share2.png'), "share2")
        self.display_image(os.path.join(self.output_folder, 'reconstructed_image.png'), "reconstructed")

    def decrypt_image(self):
        if not self.output_folder:
            messagebox.showerror("Error", "Please process an image first.")
            return

        share1_path = os.path.join(self.output_folder, 'share1.png')
        share2_path = os.path.join(self.output_folder, 'share2.png')
        output_path = os.path.join(self.output_folder, 'decrypted_image.png')

        decrypted_image = reverse_xor_decryption(cv2.imread(share1_path, cv2.IMREAD_GRAYSCALE),
                                                 cv2.imread(share2_path, cv2.IMREAD_GRAYSCALE))

        cv2.imwrite(output_path, decrypted_image)
        messagebox.showinfo("Success", f"Decrypted image saved as '{output_path}'.")
        self.status_label.config(text="Status: Decryption Complete!", fg="#00ff00")

        self.display_image(output_path, "decrypted")


if __name__ == "__main__":
    root = tk.Tk()
    app = CreativeVisualCryptographyApp(root)
    root.mainloop()



