This project develops a binary classification model to detect money mule accounts for MyInvestor, a Spanish digital investment platform. A mule account is defined as one used to receive and transfer illegally obtained funds, and its detection is a critical component of any modern anti-money laundering (AML) compliance framework.

The model will be trained on a dataset of 1,000 accounts, combining account opening information with transactional behaviour recorded throughout the lifetime of each account. The available features include device information, transaction amounts, declared profession, and stated income.

Given the sensitivity of the use case and the regulatory environment in which MyInvestor operates, the chosen models prioritise interpretability and auditability alongside predictive performance. The output of the model is a risk score that can be passed to a compliance team for review, rather than an automated decision.
