import re

path = r"d:\Music Recommendation\app.py"
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
out = []

for i, line in enumerate(lines):
    if 'st.subheader("🎶 Recommendations")' in line:
        # insert wrapper start before this line
        out.append('        st.markdown("<div class=\"results-container\">", unsafe_allow_html=True)\n')
        out.append(line)
        continue
    out.append(line)

# Now add closing div after the dataframe block
# We will locate the line containing "st.dataframe" that uses results_df and add closing right after
new_out = []
for line in out:
    new_out.append(line)
    if 'st.dataframe(' in line and 'results_df' in line:
        new_out.append('        st.markdown("</div>", unsafe_allow_html=True)\n')

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_out)

print('Wrapped recommendation section with results-container div')