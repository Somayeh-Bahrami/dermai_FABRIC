# Project Guidelines

## Writing style

- No em dashes anywhere.

## Code style

- Minimize comments. Write code that explains itself through clear, obvious names for variables, functions, and classes so any person or agent immediately understands what each construct does. Add a comment only when it genuinely improves understanding of something the code cannot convey on its own.

- Always strive for simplicity. Keep code concise, modular, and reusable, as simple as it can be without ever compromising correctness. Favor reusable code, remove duplication, and actively look for commonalities across workflows so shared logic can be extracted in a predictable and obvious way.

## Course Project Requirements (CS7643 Project Guidelines, Zsolt Kira, Summer 2026)

The following is the full course project guidelines document, included verbatim so agents always know the project requirements.

CS7643: Project Guidelines
Instructor: Zsolt Kira
Summer 2026

1 Project Details

This project provides an opportunity for you to: (1) gain experience implementing deep models and (2) try Deep Learning on problems that interest your team. Team sizes can be up to four students; however, we recommend groups of three. The amount of effort should be at the level of one homework assignment per group member. A self-contained report of your project will be the sole deliverable. After the class, we will post all final reports so that you can learn about each other's work. Additionally, we will allow people to upload additional code, videos, and other supplementary material as a zip file (similar to the code upload done for assignments).

1.1 Final Report Template

Your final write-up is required to be between 4-6 pages and structured like a paper from a computer vision conference (CVPR, ECCV, ICCV, etc.). To facilitate this, please use the provided Latex Template (https://www.overleaf.com/read/fdjpfsdhztfp). The template standardizes formatting, e.g., font size and margins, and allows us to fairly judge all student projects. The final report must be submitted in PDF format and should completely address all points outlined in the rubric that follows. Please note, supplementary material linked to in the final report is not guaranteed to be used for evaluation of this project.

1.2 Rubric (60 points)

We are not looking to see if you succeeded or failed at accomplishing what you set out to do. Its OK if your results are not good. What matters is that you gain an in-depth understanding of Deep Learning as it relates to your project, and can clearly communicate that understanding through your analysis. Note that you must justify your design, implementation, and experimentation decisions using your knowledge and data. You should make claims about why you think the results turned out the way they did and perform specific experimentation (or gather relevant data) to justify your claims. A former DARPA director, George H. Heilmeier, came up with a list of questions for evaluating research projects. We have adapted that list for our rubric.

1.3 Introduction / Background / Motivation

- (5 points) What did you try to do? What problem did you try to solve? Articulate your objectives using absolutely no jargon.
- (5 points) How is it done today, and what are the limits of current practice?
- (5 points) Who cares? If you are successful, what difference will it make?
- (5 points) What data did you use? Provide details about your data, specifically choose the most important aspects of your data mentioned here: Datasheets for Datasets (https://arxiv.org/abs/1803.09010). Note that you do not have to choose all of them, just the most relevant.

1.4 Approach

- (10 points) What did you do exactly? How did you solve the problem? Why did you think it would be successful? Is anything new in your approach?
- (5 points) What problems did you anticipate? What problems did you encounter? Did the very first thing you tried work?

1.5 Experiments and Results

- (10 points) How did you measure success? What experiments were used? What were the results, both quantitative and qualitative? Did you succeed? Did you fail? Why? Justify your reasons with arguments supported by evidence and data. Make sure to mention any code repositories and/or resources that you used!

1.6 Additional

- (5 points) Appropriate use of figures / tables / visualizations. Are the ideas presented with appropriate illustrations? Are the results presented clearly; are the important differences illustrated?
- (5 points) Overall clarity. Is the manuscript self-contained? Can a peer who has also taken Deep Learning understands all of the points addressed above? Is sufficient detail provided?
- (5 points) Finally, points will be distributed based on your understanding of how your project relates to Deep Learning. Here are some questions to think about:
  - What was the structure of your problem? How did the structure of your model reflect the structure of your problem?
  - What parts of your model had learned parameters (e.g., convolution layers) and what parts did not (e.g., post-processing classifier probabilities into decisions)?
  - What representations of input and output did the neural network expect? How was the data pre/post-processed?
  - What was the loss function?
  - Did the model overfit? How well did the approach generalize?
  - What hyper-parameters did the model have? How were they chosen? How did they affect performance? What optimizer was used?
  - What Deep Learning framework did you use?
  - What existing code or models did you start with and how did these starting points help?

Note that at least some of these questions and others should be relevant to your project and should be addressed in the PDF. You do not need to address all of them in full detail. Some may be irrelevant to your project and others may be standard and thus require only a brief mention. For example, it is sufficient to simply mention the cross-entropy loss was used and not provide a full description of what that is. Generally, provide enough detail such that someone with an appropriate background (in both Deep Learning and your domain of choice) could replicate the main parts of your project somewhat accurately.

2 Team Contributions

Your report must include a table with a row entry for each team member that provides a summary of what that member contributed to the project. All team members should contribute to the technical portion, e.g. implementation, experimentation, analysis, etc. and not just contribute to things like report writing, etc. Note that this section does not count towards your page limit.

3 References

Your report must include citations for relevant work including papers that inspired you or that you re-implemented, etc. The papers you cite in this section should be referenced within the paper itself. Note that this section does not count towards your page limit, so please be comprehensive.

4 Submission

4.1 Final Report

When the final report is complete and your team is ready to submit, choose one person to upload a PDF version of the report to the Final Project group assignment on Gradescope. The submission process is similar to that of other assignment uploads to Gradescope; however, you must add all group members before completing your submission. For more information, please see Adding Group Members (https://help.gradescope.com/article/m5qz2xsnjy-student-add-group-members).

4.2 Supplementary material

Supplementary material can be uploaded as a zip file to the Final Project (Supplementary Material) assignment in Gradescope. Follow the same process as specified for the Final Report and be sure to include all group members.
