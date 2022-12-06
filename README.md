# Projeto House Rocket - Empresa fictícia de compra e venda de imóveis
Este projeto tem por objetivo auxiliar o time de negócio da empresa House Rocket a tomar decisões embasadas em dados.

Projeto publicado na URL: https://nickoboco-house-rocket-insights-streamlit-app-zj8dfp.streamlit.app/

## Questão de negócio
A empresa House Rocket decidiu investir na região de Seattle e o CEO da companhia fez alguns quesitonamentos ao time de dados com relação a isso. Com isso em mente, o time de dados têm agora a responsabilidade de realizar uma análise do mercado imobiliário dessa região. Portanto, este projeto tem por objetivo identificar insights que apoiem nas tomadas de decisões, respondendo as perguntas do CEO e maximizando o lucro da empresa.

## Premissas do negócio
- As condições dos imóveis [condition] foram traduzidas da seguinte forma: 1=bad, 2 e 3=regular, 4=good, 5=great
- A recomendação de compra foi feita levando em conta a estação do ano (EUA) e condição do imóvel
- A recomendação de venda foi estimada em 30% acima do valor de compra [price] desde que fique abaixo do preço médio de venda na região [zip code]

## Planejamento da solução
- Realizar a coleta dos dados em um repositório público
- Fazer a limpeza e análise exploratória utilizando Python
- Contruir uma solução de dados no Streamlit
- Publicar a solução no Streamlit Cloud

## Insights
Os principais insights encontrados foram:
### 1 - Imóveis que possuem vista para água são em média 30% mais caros que os demais
- Verdadeiro! Imóveis com vista para água são em média 213% mais caros que os demais
### 2 - Imóveis renovados são em média 15% mais caros que os demais
- Verdadeiro! Imóveis renovados são em média 30% mais caros que os demais
### 3 - Imóveis com vista para água ficam em média 20% mais caros durante o verão
- Verdadeiro! Imóveis com vista para água ficam até 14% mais caros no verão
### 4 - Imóveis construídos até 1955 são em média 50% mais baratos que os demais
- Falso! A diferença é somente de 1%

## Resultados
Caso a House Rocket realize a compra dos imóveis indicados pelo time de dados, o investimento seria de $77,541,432.00 com lucro de $23,262,429.60.

## Conclusão
O objetivo desse projeto era prover insights valiosos ao time de negócio e auxiliá-los na tomada de decisão, portanto foi atingido. 

## Próximos passos
Refinar as sugestões de compra considerando imóveis que possuem preço abaixo do mercado de acordo com as caracteristicas.
