# Efficient Handwriting Generation from ASCII Text with c-LSGANs
## Amogh Shreedhar Inamdar, Anagha M Rajeev, Sindhuja V Rai

A system for efficiently generating large amounts of handwritten text from ASCII input, using conditional Least-Squares Generative Adversarial Networks (c-LSGANs) to generate images of input characters and a heuristic to stitch character images into a page of handwriting.

Input ASCII text is processed character-by-character. Each character of the text is encoded and used to generate a corresponding image of that character. These images are stitched together to form a handwritten version of the input text.

We have also developed a simple handwriting generation web app. Input text is captured and a Flask backend interfaces with our Python module to generate handwriting (using Keras over Tensorflow).
