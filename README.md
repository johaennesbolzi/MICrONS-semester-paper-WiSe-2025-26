# MICrONS-semester-paper-WiSe-2025-26

Dear users,

with this ultimate guide to successfully handling MICrONS, I am trying to make this project more accessible for unexperienced and/or interested people that want to work with the largest connectome to date.
Please note that I am an untrained medical student who did not have any prior experience with coding and programming. I learned the Python basics just for this project and my knowledge is far from perfect. So, in case you have used Python or other languages before, you might find my code too complicated or even find mistakes.


Setup

For starters, you will need access to a computer with its administration rights. We will be installing different applications and programs for easy access, so you need to have permission for your device to continue.

1. Preparation

We will use Anaconda to set up an environment that allows us to install different versions of Python and its libraries without interfering with other projects you might have going on. If you do not want to do this, skip this step.
Since there are great tutorials out there on YouTube that I have used myself, I will not explain this in detail. Just follow the steps and install the applications.
https://www.youtube.com/watch?v=4DQGBQMvwZo

Once you have done this, open GitBash and enter:

	conda create --name INSERT A NAME HERE python=3.10.19

Python 3.10.19 is a good version for MICrONS which I have used myself

	conda activate mein_env
	conda install ipykernel
	jupyter lab
  
This will open up jupyter lab but you can use a coding platform of your choice.


2. CAVEclient

We now have to install a token. The following steps are copied from https://tutorial.microns-explorer.org/
Open GitBash within the environment and enter

	pip install caveclient

Then open jupyter lab and create a notebook and enter the following

	from caveclient import CAVEclient
 	client = CAVEclient()
 	client.auth.setup_token(make_new=True)
  
This will open a new browser window and ask you to log in.
You will show you a web page with an alphanumeric string that is your token.
Copy your token, and save it to your computer using the following:

  	client.auth.save_token(token=YOUR_TOKEN)
  
To check if your setup works, run the following:

  	client = CAVEclient('minnie65_public')
  
If you don’t get any errors, your setup has worked and you can move on!

You are now ready to work with MICrONS!
Most tools are available through https://tutorial.microns-explorer.org/ which offers exact documentation of how to access the data and extract the information that you need.
