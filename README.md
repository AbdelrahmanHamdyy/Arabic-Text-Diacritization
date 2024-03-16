<div align= >

#  <img align="center" height="75px"  src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExOTJ6YXI3M2l4OW8xMTd1NmJlM2E2cXl6Mmczdnc5cnE0YnI4OWNyaSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/iJJNCuhOVeDXxoKiAU/giphy.gif"> Arabic Text Diacritization


</div>
<div align="center">
   <img align="center" height="350px"  src="https://cdn.dribbble.com/users/1092177/screenshots/2649569/dribbble.gif" alt="logo">
   <br>

   ### ”وَعَلَّمَكَ مَا لَمْ تَكُنْ تَعْلَمُ“
</div>

<p align="center"> 
    <br> 
</p>

## 📝 Table of Contents

- <a href ="#about"> 📙 Overview</a>
- <a href ="#started"> 💻 Get Started</a>
- <a href ="#modules">🤖  Modules</a>
    - <a href="#preprocessing">🔁 Preprocessing Module</a>
    - <a href="#feature">💪 Feature Extraction Module</a>
    - <a href="#selection">✅ Model Selection</a>
- <a href ="#contributors"> ✨ Contributors</a>
- <a href ="#license"> 🔒 License</a>
<hr style="background-color: #4b4c60"></hr>

<a id = "about"></a>

## 📙 Overview

<ul>
<li> Arabic is one of the most spoken languages around the globe. Although the use of
Arabic increased on the Internet, the Arabic NLP community is lagging compared to
other languages. One of the aspects that differentiate Arabic is diacritics. Diacritics are
short vowels with a constant length that are spoken but usually omitted from Arabic text
as Arabic speakers usually can infer it easily. </li>
<li> The same word in the Arabic language
can have different meanings and different pronunciations based on how it is diacritized.
Getting back these diacritics in the text is very useful in many NLP systems like Text To
Speech (TTS) systems and machine translation as diacritics removes ambiguity in both
pronunciation and meaning. Here is an example of Arabic text diacritization:</li>
<br>
<table>
<tr>
<th>Real input</th>
<th>Golden Output</th>
</tr>
<tr>
<td>
ذهب علي إلى الشاطئ
</td>
<td>
ذَهَبَ عَلِي إِلَى اَلشَّاطِئِ
</td>
</tr>

</table>

<li> Built using <a href="https://docs.python.org/3/">Python</a>.</li>
<li>You can view
<a href="https://github.com/AbdelrahmanHamdyy/Arabic-Text-Diacritization/tree/main/dataset">Data Set</a> which was used to train the model</li>
<li>Project <a href="https://github.com/AbdelrahmanHamdyy/Arabic-Text-Diacritization/tree/main/Report.pdf"> Report</a></li>
</ul>
<hr style="background-color: #4b4c60"></hr>
<a id = "Started"></a>

## 🚀 How To Run

- First install the  <a href="https://github.com/AbdelrahmanHamdyy/Arabic-Text-Diacritization/blob/main/requirements.txt">needed packages</a>.</li> 

```sh
pip install -r requirements.txt
```

- Folder Structure

```sh
├───dataset
├───src
│   ├──utils
│   ├──constants.py
│   ├──evaluation.py
│   ├──featureExtraction.py
│   ├──models.py
│   ├──preprocessing.py
│   └──train.py
├───trained_models
├───requirements.txt
...
```

- Navigate to the src directory

```sh
cd src
```

- Run the `train.py` file to train model

```sh
python train.py
```

- Run the `evaluation.py` file to test model

```sh
python evaluation.py
```

- Features are saved in `../trained_models`


<hr style="background-color: #4b4c60"></hr>
<a id = "Modules"></a>

## 🤖 Modules
<a id = "Preprocessing"></a>

### <img align= center width=50px src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzVheHZucHh0YnVlMHdyNHFjMTJ3NnAxNnlyYW0wNW40Ynh0a2UycCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/l0u897CIX2Z5Qpnxcr/giphy.gif">Preprocessing Module
<ol>
<li> <strong>Data Cleaning:</strong> First step is always to clean the sentences we read from the 
corpus by defining the basic Arabic letters with all different formats of them 
they encountered 36 unique letter, the main diacritics we have in the language 
and they encountered 15 unique diacritic, all punctuations and white spaces. 
Anything other than the mentioned characters gets filtered out.</li>
<li><strong>Tokenization:</strong> The way we found yielding the best result is to divide the corpus 
into sentences of fixed size (A window we set with length of 1000) which means 
that if a sentence exceeds the window size we will go backward until the first 
space we will face then this will be the cutting edge of the first sentence, and 
the splitted word will be the first one in the next sentence and keep going like 
this. If the sentence length is less than the window size then we pad the rest of 
the empty size to ensure they’re all almost equal.</li>
<li><strong>Encoding:</strong> The last step is to encode each character and diacritic to a specific index which 
is defined in our character to index and diacritic to index dictionaries. Basically 
transforming letters and diacritics into a numerical form to be input to our.</li>
<li><strong>Failed Trails:</strong> We Tried not to give it a full sentence but a small sliding window 
and this sliding window in flexible in size as we can determine the size of 
previous words we want to get and the size of the next words.</li>
</ol>
<a id = "Feature"></a>

### <img align= center height=60px src="https://media0.giphy.com/media/fw9KH5k7W2BVb78Wkq/200w.webp?cid=ecf05e472gayvziprwm50vr429mjzkk6lic31u4tegu821k7&ep=v1_stickers_search&rid=200w.webp&ct=s">Feature Extraction Module


<ol>
<li><strong>Trainable Embeddings(which we used): </strong> Here we use the Embedding layer provided by torch.nn. which gives us trainable embeddings on the character level. This layer in a neural network is responsible for transforming discrete 
input elements, in our case character indices, into continuous vector 
representations where each unique input element is associated with a 
learnable vector and these vectors capture semantic relationships 
between the elements. </li>
<li><strong> <a href="https://github.com/bakrianoo/aravec">AraVec</a> CBOW Word Embeddings: </strong> AraVec is an open-source project that offers pre-trained distributed word 
representations, specifically designed for the Arabic natural language 
processing (NLP) research community. </li>
<li><strong> <a href="https://github.com/aub-mind/arabert">AraBERT</a>: </strong> Pre-trained Transformers for Arabic Language Understanding and Generation (Arabic BERT, Arabic GPT2, Arabic ELECTRA)</li>
</ol>
The next approaches weren’t possible on pure Arabic letters because these 
libraries tokenize on English statements. They expect the data to be joined 
sentences in English form so we had to find a way to deal with this issue. After a 
bit of research, we found a method that basically maps Arabic letters and 
diacritics to English letters and symbols by using <a href="https://en.wikipedia.org/wiki/Buckwalter_transliteration">Buckwalter</a> transliterate and 
untransiliterate functions, we were able to switch the language for the feature 
extraction by ourselves part.
<ul>
<li><strong>Bag Of Words: </strong> by using the CountVectorizer method which is trained on the 
whole corpus. The vectorizer gets the feature names after fitting the data and 
then we save them to a csv file which represents the bag of words model. We 
can index any word and get its corresponding vector which describes its count in 
the corpus and no info about the position of this word.</li>
<li><strong>TF-IDF: </strong> The TfIdfVectorizer initializes our model and we choose to turn off 
lowercasing the words. After transforming and fitting the model on the 
input data, we extract the feature names out and this will be out words 
set that we’ll place them in column headings of each column in the 
output csv file.</li>
</ul>

<a id = "Selection"></a>

### <img align= center height=60px src="https://media0.giphy.com/media/YqJxBFX7cOPQSFO6gv/200w.webp?cid=ecf05e47q2pctv46mon3iqculvvgg8k8bruy7d5or1kf1jh8&ep=v1_stickers_search&rid=200w.webp&ct=s">Model Selection

Fitting training data and labels into an <strong>5-Layer Bidirectional LSTM</strong> which gives us 97% accuracy.



<hr style="background-color: #4b4c60"></hr>

<a id ="Contributors"></a>

## 👑 Contributors 

<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/AbdelrahmanHamdyy"><img src="https://avatars.githubusercontent.com/u/67989900?v=4" width="150;" alt=""/><br /><sub><b>Abdelrahman Hamdy</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/BeshoyMorad" ><img src="https://avatars.githubusercontent.com/u/82404564?v=4" width="150;" alt=""/><br /><sub><b>Beshoy Morad</b></sub></a><br />
    </td>
       <td align="center"><a href="https://github.com/AbdelrahmanNoaman"><img src="https://avatars.githubusercontent.com/u/76150639?v=4" width="150;" alt=""/><br /><sub><b>Abdelrahman Noaman</b></sub></a><br /></td>
     <td align="center"><a href="https://github.com/EslamAsHhraf"><img src="https://avatars.githubusercontent.com/u/71986226?v=4" width="150;" alt=""/><br /><sub><b>Eslam Ashraf</b></sub></a><br /></td>
  </tr>
</table>



<a id ="License"></a>

## 🔒 License

> **Note**: This software is licensed under MIT License, See [License](https://github.com/AbdelrahmanHamdyy/Arabic-Text-Diacritization/blob/main/LICENSE) for more information ©AbdelrahmanHamdyy.
