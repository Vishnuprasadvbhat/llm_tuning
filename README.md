## [Compute cost and requirement](https://blog.eleuther.ai/transformer-math/)

### [Lora,Qlora](https://arxiv.org/pdf/1902.00751)


### result before finetuning:
model name : 'distillgpt2'
tokenizer : 'distilgpt2'

input : Why can camels survive for long without water?

output : The answer is yes. If the sun's rays are shining, it will not be too hot to see them in their own eyes and therefore won't cause any damage or injury from these things that might happen during a day of sunshine on your face (e-mail me at: mamfjoe@gmail)

### result after finetuning: 
####  This result is after 1 epoch 
model size is similar: we didn't compress the model but we just finetuned:
model name : 'distillgpt2-ft'

input : Why can camels survive for long without water?

output: The camels are very thin and have a lot of moisture. They need to be kept in a dry environment. Some people like camels because they do not have enough water. They also need to keep their food and water.

actual ouput: Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.

# here We can see that model is using its previous weights and its hallucinating 

4 epoch and temp = 0.5

Why can camels survive for long without water? | Caramelization is the process by which a tiny plaque of oil comes from a tiny plaque that has bubbles and then rubs it out. The bubbles are then spread over time, and they become sticky and sticky. When they get wet you will also peel them and put them in your dish.

4 epochs and temp = 0.1
Why can camels survive for long without water? 

Caramel is the primary color of paint on the windshields when it comes to painting.  It's also a popular color for clothing, such as cotton lawns and velvet jackets.



  Why is this happening?
  Overgeneralization: LLMs are trained on vast datasets, but they do not "understand" facts the way humans do. They predict the next word based on statistical patterns rather than knowledge verification.

  Lack of Fact-Checking: LLMs lack internal mechanisms to verify the accuracy of their generated outputs. They rely on patterns in the data they were trained on, and if the data contains inconsistencies, outdated, or fictional information, it can lead to hallucination.

  Ambiguity in Input: Sometimes, LLMs generate incorrect or non-specific responses when the question is open-ended or not clear, though this isn't the case in your example.



### After 10epochs worked ryt on traing set 
Why can camels survive for long without water? | 

Camels use the fat in their humps to keep them filled with energy and hydration so they cannot be outdoors longer than in a boat. 

# Overfiiting for unseen data 
how can start learning swiming? | 

Swimming requires a constant change in posture. For instance, replace the daily stand up, then follow that for 30 minutes. This will allow you to practice all 3 steps forward.

Choose the right position on your own. For example, if you're riding a bike with 80 minutes of exercise, and be able to use 15 minutes.

Build the next Saturday morning (after a long walk) by 20 minutes. Avoid injury to cause damage to your joints. Keep gradually gain weight.