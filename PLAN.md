# The plan

This is the plan I would follow for this assignment. As in Assignment 1, give
it to your coding agent one step at a time, not all at once, and read each
step yourself first.

Each step says what to ask for and how you know it worked. A step is not done
because the agent says it is done. It is done when the check passes.

In this assignment nobody writes a program from nothing. The training code
works from the start, and every step changes one thing in it. That makes the
way you ask matter more than in Assignment 1. A good request names one
change, says what stays the same, and gives the change a name (its tag):

```
In track_a/recipe.py, draw every city equally often, and change nothing
else. Call it balanced and train it.
```

When something fails, give the agent the error text as it is, not your summary
of it.

---

## Step 0. Set up

Give Cline the request under "Getting started" in README.md.

Check: the agent ran `official.py --quick`, it ended with "validation score",
and the agent told you how long one full training takes on your computer and
gave you a link that opens your notebook on Colab. Keep the link. You filled
in the registration form on Cyber Campus.

## Step 1. Guess the cities yourself

Not a coding step. In class we guess the city of 40 validation photos
together (README, Problem 1). Write down your score and the clues you used.

## Step 2. Train the starter recipe twice

```
Train the starter recipe with seeds 0 and 1, with the tag baseline.
```

If one run takes more than about 15 minutes on your computer, the agent
prepares the runs for Colab instead. Then you open your Colab link, choose
Runtime, Run all, wait until your browser downloads `from_colab.zip`, and tell
the agent. README, "Google Colab", says what Colab asks you in every new
session.

Check: `experiments.csv` has two rows with the tag baseline and validation
scores near 0.45, and `LOG.md` has an entry for them. The two scores differ a
little although only the seed differs: that is how much a score moves by
chance.

## Step 3. Look at the mistakes

```
Show me the mistakes of the baseline.
```

The agent makes sheets of the validation photos the model got most
confidently wrong, one city at a time. Look at every sheet.

Check: you wrote in `LOG.md`, in your own words, two things the mistakes have
in common.

## Step 4. Your first release

```
Release my Track A recipe. The title is "starter recipe".
```

Check: the agent ran `official.py --quick` first, and `track-a-1` is on your
repository's Releases page. The next morning after 6:00 it is on the
leaderboard.

## Step 5. The mid-point check (30 October, 22:00)

Make one change of your own choosing (Step 6 says how), so that `LOG.md`
has at least three runs. End `LOG.md` with a short plan: what your model gets
wrong, what you will try next, and where you will find more photos for Track B.

Check: the three items under Problem 2 in README.md.

## Step 6. One change at a time (Problem 3)

Choose a change. README, Problem 3, lists some directions; you can also ask
the agent what a change would do before you decide. Then ask for it as in the
example at the top. Before it trains, the agent shows you what it changed.

Check, for every change:
- the change is exactly the one you asked for, and nothing else changed
- an entry in `LOG.md` with the score and your reason

Do this at least four times, and keep what helps.

## Step 7. Your judgement in the loop (Problem 4)

Do this step yourself. The agent runs the tools and makes the changes you ask
for; it does not choose photos.

1. Look at the mistakes (Step 3) or at the training photos the model doubts
   most:
   ```
   Show me the suspects of balanced.
   ```
2. Write down, before you change anything, what you see and the rule you
   will follow. For example: "drop training photos taken indoors".
3. Name the photos by sheet and number:
   ```
   Drop photos 5, 17 and 40 from suspects_busan_00, reason "indoors".
   ```
4. Train the new selection and compare.

Check: `track_a/dropped.csv` lists every photo you dropped with your reason,
and `LOG.md` has the scores before and after. Keep the exact requests you gave
the agent in this step, for the report.

## Step 8. Track B, the first model

```
Train train_b.py as it is and make it my Track B submission.
```

Then release it, as in Step 4, as `track-b-1`.

Check: `check_submission.py` printed "format and limits: OK", and you have
the score on the validation photos (and on `data/sample_target`, once those
photos are out).

## Step 9. Track B, better (Problem 5)

One change at a time, as in Step 6: a different model or image size, the
changes that helped in Track A, more photos. For your own phone photos:

```
Sort the photos in <the folder> into the cities with gps_sort.py.
```

Check: the model still passes `check_submission.py`, and `LOG.md` has an
entry for each change.

## Step 10. Release as you go

Release whenever you have something better (Step 4). Only your newest release
of each track before 6:00 counts each morning. Compare your validation score
with your leaderboard score; when they disagree, write down why you think
they do.

## Step 11. The long report

Go through the "Write this down" boxes in README.md in order, with your agent.
It fills in `long_report/long_report.tex` from `experiments.csv`, `LOG.md` and
the saved output of your runs.

Check: no red text is left in the PDF, and every number in it is one that
code printed.

## Step 12. The short report

Close the agent. One page, by yourself, in Korean if that is your first
language (README, "The short report").
