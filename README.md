# Mlops-Eindopdracht Tibe Demeulemeester

## Table of Contents

- [Some context](#some-context)
  - [The chosen dataset](#the-chosen-dataset)
  - [The chosen AI model](#the-chosen-ai-model)
  - [Preprocessing steps](#preprocessing-steps)
- [Setting up](#setting-up)
  - [GitHub Secrets](#github-secrets)

## Some context

### The chosen dataset

I have chosen for a dataset about emotion sentiment were you have 3 column: **tweet_id** the unique id of the tweet,  **sentiment** were you can see what emotion the text was and **content** the text itself.

### The chosen AI model

The chosen AI model I have chosen is a **Bidirectional LSTM** model with using a preexisting **Glove** Embedding.

### Preprocessing steps

The preprocessing consists of the following steps: 

1. Removing the column **tweet_id**
2. Removing the rows were **content** is the same but has a different sentiment
3. Normalizing the text, which includes:
   - Setting everything to lower case
   - Removing stop words
   - Removing numbers
   - Removing punctuation
   - Removing URLs
   - Lemmatization

## Setting up

### GitHub Secrets 

Firstly you will need to get **AZURE_CREDENTIALS** this will be used to log into the azure CLI you can do this by running the following CLI command.

```bash
az ad sp create-for-rbac --name "myApp" --role contributor --scopes /subscriptions/{subscription-id}/resourceGroups/{resource-group} --json-auth 
```

You will get a JSON that looks like this.

```text
{
    "appId": "<appId>",
    "displayName": "<displayName>",
    "name": "<name>",
    "password": "<password>",
    "tenant": "<subID>"
}
```

Use this as the GitHub Secret for **AZURE_CREDENTIALS**.

Secondly you will need to get the **TOKEN** secret which will be used to login into the **GitHub container registry** 


To get the token you will first have to create a **Personal access token** you can do that by going to your **GitHub settings** then clicking on **developer settings**. After that going to **Personal access token** then **Token (classic)**. Now you are on the page where you can see all your Personal access tokens. you can use an existing one or create a now one by clicking on **Generate new token** again choose the classic version. 

![Image of github Token](images/githubToken.png)

Choose a name for your token and make sure your token has acces to the **repo** section and **write:packages** section now create your token.

Now use the token you have been given as the Repository secret **TOKEN**

![Image of Repository secrets](images/RepositorySecrets.png)

### Azure Machine Learning

#### Uploading data to your Azure Machine Learning studio

the dataset you can find here [emotions.csv](dataset/emotions.csv) and the needed word vectors you can find on this website [Glove website link](https://nlp.stanford.edu/projects/glove/) 

![Glove word vectors image](images/glove.png)

download the zip file the only file you need from the zip is the **glove.6B.200d.txt** file 

Now go to the Machine Learning Studio and go to **Data** then click on **Create** then you have something like this. 
![Data asset](images/data.png)

Now choose a Name I picked **DataFolder** as my name and use type **Folder (uri_folder)** then click next.
Then choose **From local files** and click next untill you can upload your folder with the dataset file and glove file.

