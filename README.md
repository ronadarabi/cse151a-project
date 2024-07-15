# CSE 151A Group Project

## Collaborators 
* Justin 
* Rona 
* Diego
* Jose
* Logan  
* Daniel

## Preprocessing Data 

The dataset chosen has been preprocessed into LAB* format

Initially, python libraries such as NumPy, Matplotlib, scikit-image are installed and imported to set up the environment. The data is then downloaded from google drive and unzipped. The data is initially split into l, ab1, ab2, and ab3, so we stack ab1, ab2, and ab3 to then concatenate with l values in order to form the correct LAB values of each image. We then reshaped, and loaded the array into a dataframe in order to easily analyze, naming the columns after the corresponding component of the LAB naming scheme. Once we finish preprocessing the data, we then proceed to analyze it. We use the data frame’s built-in describe feature, visualize the images using only L values, L and A values, L and B values, as well as all three. We then plot the distribution charts of the density of the values for each component of the LAB color space, as well as the density of the values for each component of the RGB color space.