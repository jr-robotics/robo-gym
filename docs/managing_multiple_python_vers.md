# Managing Multiple Python Versions


To manage multiple Python versions on the same machine we suggest to use [pyenv](https://github.com/pyenv/pyenv).

[Here](https://realpython.com/intro-to-pyenv/) there is a great article on pyenv.

We recommend using [pyenv-installer](https://github.com/pyenv/pyenv-installer) to install pyenv on your machine.

Once pyenv is installed you can check the python versions available with:

```
pyenv versions
```

If this is the first time that you use pyenv you the previous command should return:

```
* system (set by /home/username/.pyenv/version)
```
Which shows that the only python installation available is the system default python.

The Standard Installation of the Robot Server Side has to be performed on
the system default python which should be python 2.7.

The Environment Side installation requires Python >= 3.5, let's first install a suitable python version (e.g. 3.6.10) with:

```
pyenv install 3.6.10
```

Now we can create a virtual environment in which to install and run robo-gym:

```
pyenv virtualenv 3.6.10 robo-gym
```

Where 3.6.10 is the desired python version and robo-gym is the name of the virtual
environment.

To activate the virtual environment use:

```
pyenv activate robo-gym
```

You should see in your shell something like:

```
(robo-gym) user@machine:$
```

This means that you are now within the virtual environment.
This is the virtual environment in which you should perform the Environment Side
installation. When you run `pip install robo-gym` the robo-gym package will
be installed in the virtual environment.
