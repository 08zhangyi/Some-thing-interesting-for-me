import numpy as np
from PIL import  Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

with open('hound.txt') as infile:
    text = infile.read()

mask = np.array(Image.open('holmes.png'))

stopwords = STOPWORDS
stopwords.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may', 'littile', 'say', 'must', 'way', 'long',
                  'yet', 'mean', 'put', 'seem', 'asked', 'made', 'half', 'much', 'certainly', 'might', 'came'])

wc = WordCloud(max_words=500,
               relative_scaling=0.5,
               mask=mask,
               background_color='white',
               stopwords=stopwords,
               margin=2,
               random_state=7,
               contour_width=2,
               contour_color='brown',
               colormap='copper').generate(text)

colors = wc.to_array()

plt.figure()
plt.title("Chamberlain Hunt Academy Senior Class Presents:\n", fontsize=15, color='brown')
plt.text(-10, 0, "The Hound of the Baskervilles", fontsize=20, fontweight='bold', color='brown')
plt.suptitle("7:00 pm May 10-12 McComb Auditorium", x=0.52, y=0.095, fontsize=15, color='brown')
plt.imshow(colors, interpolation="bilinear")
plt.axis('off')
plt.show()
# plt.savefig('hound_wordcloud.png')