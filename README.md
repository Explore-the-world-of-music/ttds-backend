# TTDS Backend Setup
## Downloading the repo

Pull the repo from GitHub (for mac/linux): 
```
mkdir ttds-backend
cd ttds-banckend
git init
git remote add origin https://github.com/Explore-the-world-of-music/ttds-backend.git
git pull origin master
```
	
Download **.password** from Discord and put it in root (where app&#46;py, README&#46;md, etc. are). Send a message on Discord if you can't find the file.

There are two lines in this file. The server reads the **FIRST** line and uses it to connect to the database. 
  
By default, that will be the remote Heroku database.

If you want to connect to the local database instead, **SWAP** the two lines. 

## Configuring Flask
[Useful tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)

Create a new virtual enviroment (optional, you can also make a Conda enviroment instead) 
```
python3 -m venv venv
```
Then, activate it:
```
source venv/bin/activate (mac/linux)
or
venv\Scripts\activate (windows)
```
Install all requirements using the following command:
```
pip install -r requirements.txt
```

You can run the server using "**flask run**" from the directory with the app&#46;py file. It will automatically connect to the **remote** database using the password file.
- To make a sample get request, go to localhost:5000/api/song?query=Give%20You%20Up
- It should return a JSON file with results

- If you fancy, you can use [Postman](https://www.postman.com/) to send requests in a slightly easier format
	- Documentation : https://learning.postman.com/docs/sending-requests/requests/
	
- If you need the frontend locally, you'll need to pull the frontend repo as well in a separate folder (for example, "ttds-frontend") and follow the guide from there. You will then need two servers running: the node.js server for the frontend ("npm start") and the Flask backend server ("flask run"). They will communicate with each other automatically.

## Setting up local Postgres (**OPTIONAL**)
Tutorials: 
- [Windows](https://www.postgresqltutorial.com/install-postgresql/)
- [Linux](https://www.postgresqltutorial.com/install-postgresql-linux/)
- [Mac OS](https://www.postgresqltutorial.com/install-postgresql-linux/)

### Windows/Mac: 
1. [Download](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) Postgres.
The version that the project is using is 13.1 (should be the latest one you can download)
2. Tick all 4 components during the install (PostrgeSQL server, pgAdmin, stack builder, cli tools)
3. When asked to provide a password, type "*postgres*".
4. Keep the port as 5432
5. Install
6. If you're using MacOS, you can check if postgres is running using "which psql". For Windows, you can check your servies (start -> run -> services.msc).

### Linux and WSL
If using Linux or WSL, follow the guide mentioned above and install Postgres using apt-get. Then, you will need to change the password to "postgres": https://docs.boundlessgeo.com/suite/1.1.1/dataadmin/pgGettingStarted/firstconnect.html#setting-a-password-for-the-postgres-user

## Setting up pgAdmin
1. Get [pgAdmin](https://www.pgadmin.org/download/). Skip this step if you have installed Postgres locally because it comes with pgAdmin!
2. Launch pgAdmin. It will run in your browser. If you have local Postgres, you should automatically see a server called "localhost". 
3. Now it's time to connect to the remote database! Right click on "Servers" and select "Create" -> "Server".
4. Give a server any name (for example, "ttds-remote").
5. Click on "Connection" and fill in the following using the details from Discord:
   - Host
   - Port
   - Maintenance Database
   - Username
   - Password
   - Tick "save password"  
6. Click on "Advanced".
   - Paste the name of the database (the one starting with "dd1...") in "DB Restriction". This is done to hide databases that you don't have access to from the list.
7. Click on "Save" and expand the servers list.  
8. Go to databse_name/Schemas/public/Tables
    - You can right click and select "View/Edit data" to view the rows
	- Use "Import/Export" to upload data as a csv. You can also make backups of the talbe (either as SQL or a custom format) using the backup button.
	- Use "Query Tool" to write regular SQL 
	- You can add data manually using "View/Edit Data". Click on it, then add it to the table and commit your changes using "Save Data Changes" (4th button in the top panel, jsut under "Properties"

## Using branches, pull requests, etc.
When multiple people work on the same repo, commiting everything to Master can lead to conflicts. Instead, we can do the following:
- When you start working on a new feature, run "git checkout -b my_awesome_feature". This is your **local** branch. 
- Use "git add"/ "git commit" as usual
- Once you're done with your changes and everything works locally, you can use "git push origin my_awesome_feature". This will push the changes to GitHub to a new branch (called "my_awesome_feature"). 
- Go to github.com and you'll see a green button that suggests that you make a pull request. Go ahead and click it! If you can't see it, click on "Branches" and select yours from the list. 
- If there are no conflicts, you can merge your pull request, which will add the code from your branch to the master branch. You can find more information [here](https://guides.github.com/activities/hello-world/#:~:text=Pull%20Requests%20are%20the%20heart,merge%20them%20into%20their%20branch.&text=You%20can%20even%20open%20pull,repository%20and%20merge%20them%20yourself.).
- Don't forget to delete the merged branch on GitHub (it's the grey button with a purple icon on the left that shows up just after you click on "merge")
- You should also delete the **local** branch that you have as well. You can do it like this:
```
git checkout master  //changes the branch back to master
git fetch origin master  //gets the new version of master from github
git branch -d my_awesome_feature
```
- Double check that master is working fine! If it isn't, try figuring out why. You can push a hotfix directly to master bypassing the pull request, just run "git add"/ "git commit" and then "git push origin master".