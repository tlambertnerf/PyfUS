Install
==================
If you don't have it installed yet, download and install the `Anaconda`_ package manager.

Download the package here and uncompress it. We will mark its location on hard drive as <path_to_the_folder>.

Open 'Anaconda Prompt' and prompt:

  >>> cd <path_to_the_folder>
  >>> conda env create -f environment.yml -n foss4fus

You have now a virtual environment up and ready for using the software. Let's install it by running:

  >>> conda activate foss4fus
  >>> python setup.py bdist_wheel


Quickstart
==================

Before using the software, you need to activate the virtual environment. To do so, open Anaconda Prompt and prompt:

  >>> conda activate foss4fus


Please check our Notion page for a QuickStart guide!


.. _Anaconda: https://www.anaconda.com/download
