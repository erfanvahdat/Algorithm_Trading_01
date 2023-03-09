import requests
# import openai
open_API="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"

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.Completion.create(
  model="text-davinci-003",
  prompt="Say this is a test",
  max_tokens=7,
  temperature=0
)