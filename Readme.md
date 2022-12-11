# A repo for GANs family
## Basics about GANs
- GANs generate new data from learning the training dataset -> Fake data
- GANS also have a discriminator to inspect the fake data
- The two networks works against each other simultaneously so that finally the GANs can generate better fake data
- 2 Losses to be minimized
### In details
- Generator first generate bad fake data
- Discriminator: Fake!
- Generator generate fake data again based on the feedback
- Discriminator: Fake!
- Generator generate fake data again
- Discriminator: Not sure...
- ...
- Generator generate fake data again
- Discriminator: These are real