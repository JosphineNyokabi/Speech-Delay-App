import re

path = r"C:\Users\JOSEPHINE\Documents\Speech Delay Project\templates\index.html"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

fixes = [
    ('result.behavioural_flags.includes("low_verbal")',    'result.behavioural_flags && result.behavioural_flags.flag_low_verbal'),
    ('result.behavioural_flags.includes("early_screen")',  'result.behavioural_flags && result.behavioural_flags.flag_early_screen'),
    ('result.behavioural_flags.includes("concern_raised")','result.behavioural_flags && result.behavioural_flags.flag_concern'),
    ('result.milestone_flags.word_delayed',    'result.milestone_flags.milestone_word_delay'),
    ('result.milestone_flags.combine_delayed', 'result.milestone_flags.milestone_combine_delay'),
    ('result.milestone_flags.respond_delayed', 'result.milestone_flags.milestone_respond_delay'),
]

for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f"Fixed: {old[:50]}")
    else:
        print(f"Already fixed or not found: {old[:50]}")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Done — file saved.")
