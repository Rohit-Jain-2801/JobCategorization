# Objective
Scrape emails from an existing email address on the basis of their subject containing keywords "Thank you for applying" and categorise them into a "job" category.

# Example 
1. User applies for a job and receives a confirmation email. 
2. The subject of the email contains the keywords "Thank you for applying".
3. User applies for n number of jobs and receives n number of emails, subject containing the keywords "Thank you for applying".
4. Filter out all the emails received after applying for a job.

# Requirements before executing the code :
Navigate to see all setting from your gmail page and follow steps listed below:
1. Turn `off` the 2-step verification for your Gmail.
2. `Enable` IMAP access from setting in via Gmail.
3. Turn `on` access to less secure apps.

# Things to note:
1. When running the code for first time a google security check web page might open, click on check activity and again click and accept yes it was me. When running code for first time google might give you multiple security alert.
2. If you run this code on python IDLE password entered might me echoed, so better use the command prompt to run the code.

# Python libraries required :
```python
import imaplib, email, getpass

import re
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import io
import json
```

# Working
1. The program takes two inputs from the user: `Gmail-ID` & `Password`.
2. The code then retrieves all the mails where the subject has keyword "Thank you for applying".
3. It is followed by extraction of job role specified in the mail.
4. Then it is passed to a model which makes use of pre-trained `Word2Vec` embeddings & predicts the Job-Category (Business / Sales-Marketing / Technical / Other).

# Command to execute the code :
`python emailjobcategorizer.py`