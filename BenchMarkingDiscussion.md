Main observations from the plots were that maybe having a difference matrix would be easier to look at than side by side plots, being vary of colour blindness (do we want to just have numbers). And trying to balance how "interpretable" something is versus how objective it is, depending on what expectations we have of the audience.

 

Privacy metrics:

- Euclidean distance between synthetic record and nearest real record

- Membership attacks: can you identify if data was used in training?

 

Data utility:

- Statistical comparison stuff as above

- ML benchmarking scores. Compare the performance of real data on a standard classification problem for example with the same problem with the synthetic data.

- Hope to have similar F1 scores. Important that performance is similar between the two (synthetic shouldn't be much better or much worse).

 

Is there a dichotomy between utility and privacy? You can do amazing in one and rubbish in the other, or we suspect we can do quite well in both.

 

Some attributes are better at identifying than others. Also main problem can come from extremes/outliers. How do we present these well whilst preserving privacy?

 

Identifiers and quasi identifiers, do we want more weighting on these?

 

Maybe we want to make sure our training data is K-anonymity before synthesising. Can add an epsilon noise term to affect utility/privacy.

 

If your data meets these thresholds, then there probably won't be individuals who are extreme outliers.

https://aboutmyinfo.org/

 

"Positive deviance analysis" looking for examples of exceptions to rules

 

"Statistical disclosure control" also of interest.

The paper mentioned on disclosure control:
GRAIMATTER Green Paper: Recommendations for disclosure control of trained Machine Learning (ML) models from Trusted Research Environments (TREs) 

https://arxiv.org/pdf/2211.01656

 
