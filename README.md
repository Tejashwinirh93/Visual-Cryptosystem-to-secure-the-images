# Visual-Cryptosystem-to-secure-the-images

##Techniques used
1. Visual Cryptography:Splits the image into two shares using a chaotic sequence (logistic map), The shares are complementary for black pixels and identical for white pixels.
2. Logistic Map:Generates a chaotic sequence used to encrypt the image pixels.
3. XOR Operation:Combines two shares to reconstruct the original image.
4. Darkening Factor:Scales the pixel values of the reconstructed image for visual adjustment.
5. PSNR (Peak Signal-to-Noise Ratio):Measures image quality by comparing the original and reconstructed images.
6. Normalized Cross-Correlation (NCORR):Measures the similarity between the original and reconstructed images.
7. GUI (Tkinter):Provides a user interface for selecting images, processing them, and viewing results.
8. OpenCV:Handles image reading, processing (binarization, thresholding), and writing.
9. Error Handling:Displays success or error messages for user interaction.


##Teammates
Tejashwini R H
Vaibhav J B
Vidya R
Yuktha B

## Requirements
This program requires the Python Image Library (PIL). The easiest way to install it is using pip:

    $ pip install --upgrade Pillow


## More info
This is a visual variant of the one-time-pad algorithm which is one of the rare crypto algorithms that has been proven to be completely unbreakable. The great thing about it is that you don't need a computer to decipher the messages you receive (but you do need one to generate the ciphered messages). If you were to use this seriously, you would first generate many random secret images and share them securely with the person you want to communicate with, then later you must use one different secret image for each message you want to send. Never reuse a secret image.

Check out the Wikipedia article on [Visual Cryptography](https://en.wikipedia.org/wiki/Visual_cryptography) for more details.

Have fun!
