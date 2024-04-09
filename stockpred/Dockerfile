# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:3.12

# The enviroment variable ensures that the python output is set straight
# to the terminal with out buffering it first
ENV PYTHONUNBUFFERED 1

# Set the working directory to /credit_approval_system
WORKDIR /django

COPY requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

COPY . .
