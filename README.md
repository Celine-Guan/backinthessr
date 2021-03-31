## Recognizing human sentiment from audio and text recordings ##

# Data analysis
- **Description**: We analyzed both audio recordings and text transcripts to predict the sentiment behind a person’ sentence. Our intuition was that combining two models with two different sources, using multi-modal learning, could improve our performance.
- **Data Source**: a data set from Carnegie Melon University called [CMU-MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/) which is the largest dataset of sentence level sentiment analysis and emotion recognition in online videos. It contains more than 65 hours of annotated video from more than 1.000 speakers and 250 topics.
The data was divided between segments of variable lengths, each representing a full spoken sentence (features), and the sentiment, our target, which varied between the values -3 to 3 (from negative to positive, 0 being neutral).
- **Type of analysis**: Stacking machine learning models and deep learning models for speech sentiment analysis. 

**For more details about the data preprocessing and the machine learning and deep learning models, please see the blog [@ medium](https://medium.com/@garetcorentin/speech-sentimental-analysis-backinthessr-adf433488845)**

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for backinthessr in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/backinthessr`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "backinthessr"
git remote add origin git@github.com:{group}/backinthessr.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
backinthessr-run
```

# Install

Go to `https://github.com/{group}/backinthessr` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/backinthessr.git
cd backinthessr
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
backinthessr-run
```
