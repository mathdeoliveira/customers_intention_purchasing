# Intenção de compra de um usuário

## Descrição

A empresa de vendas de produtos de e-commerce XYZ Corp. contratou um novo time de dados onde eles estariam sido alocados em um novo projeto idealizado pelo seu fundador.
O seu desejo é de existir um sistema inteligente onde é possível realizar uma previsão, em tempo real, de que uma nova jornada do usuário no seu website existe uma intenção de compra desse visitante.

## Dados

Estamos atuando em um cenário de aprendizado e portanto já possuímos os dados extraídos e disponíveis, algo que um ambiente real seria necessário alguns passos antes de chegar nessa etapa.

Os dados consiste em 12330 sessões de usuários distintos no site, esses dados foram capturados durante o período de um ano e com isso foi possível capturar dados sem tendências de campanhas de marketing, dias de comemorações e pouca informação sobre os usuários.

Portanto teremos dados sobre durante visitas desses usuários, juntamente com algumas informações da sessão e do usuário e com isso podemos descrever sobre os nossos dados, para realizar o download [acesse esse link.](https://docs.google.com/spreadsheets/d/1HMdb5GkH3di0bFEH4CRY8k487UbToP2APplGhVbqIjE/edit#gid=986013730)

### Descrição dos dados

| Nome da coluna          | Descrição                                                                                                        |
|-------------------------|------------------------------------------------------------------------------------------------------------------|
| Administrative          | Número de páginas administrativas que o usuário visitou                                                          |
| Administrative_Duration | Total de tempo gasto na categoria da página administrativa                                                       |
| Informational           | Número de páginas informativa que o usuário visitou                                                              |
| Informational_Duration  | Total de tempo gasto na categoria da página informativa                                                          |
| ProductRelated          | Número de páginas relacionadas à produtos que o usuário visitou                                                  |
| ProductRelated_Duration | Total de tempo gasto na categoria da página relacionadas à produtos                                              |
| BounceRates             | A porcentagem de visitantes que entram no site por meio dessa página e saem sem acionar nenhuma tarefa adicional |
| ExitRates               | A porcentagem de visualizações de página no site que terminam nessa página específica.                           |
| PageValues              | O valor médio da página em média sobre o valor da página de destino e/ou a conclusão de um eCommerce             |
| SpecialDay              | Este valor representa a proximidade da data de navegação para dias ou feriados especiais                         |
| Month                   | Mês da jornada do usuário                                                                                        |
| OperatingSystems        | Indica o sistema operacional utilizado pelo usuário em sua jornada                                               |
| Browser                 | Indica o browser utilizado pelo usuário em sua jornada                                                           |
| Region                  | Indica a região de acesso onde o usário acessou o website                                                        |
| TrafficType             | Indica qual tipo do trafégo foi utilizado pelo usuário para acesso ao website                                    |
| VisitorType             | Indica se o usuário é um retornante ou novo                                                                      |
| Weekend                 | Indica se no dia do acesso do usuário foi um dia de final de semana                                              |
| Revenue                 | Resultado da navegação do usuário se foi realizado uma compra ou não                                             |


## Solução
### Planejamento
Em todo projeto de data science se inicia no planejamento para que fique claro qual é o problema e quais são as soluções para o negócio prosperar e o projeto não ter um fim trágico.
#### Produto de dados
Nada mais juntos do que iniciar o projeto tendo como suporte uma documentação com o escopo bem definido e que o problema atual esteja bem contextualizado e os possíveis riscos, para tal iremos utilizar o framework [Data Product Canvas](https://medium.com/@leandroscarvalho/data-product-canvas-a-practical-framework-for-building-high-performance-data-products-7a1717f79f0) para que fique bem organizado, para acessar o canvas do nosso projeto [Data Product Canvas - Customer Intention Purchasing](https://docs.google.com/spreadsheets/d/17nxaMoUPMBykIW7YVEtPSSQw32izteSuFG6_QoXxhdc/edit?usp=sharing).

Com esse canvas preenchido com parceria de todas as áreas/pessoas interessadas no projeto, se faz possível a continuação do projeto e também serve como documentação para consulta posteriores de manutenção e/ou melhorias. 

#### Entendendo o problema de negócio

O desejo do fundador consiste em ter uma predição de intenção de comprar de um novo visitante na sua loja, porém não diz especificamente quais seriam, por exemplo, os possíveis ganhos, qual seria a atuação em caso de ter a intenção positiva e como atualmente é feito, caso tenha, portanto em um cenário de estudo vamos realizar algumas definições hipotéticas.

**Qual é o objetivo da empresa?** Com esse sistema de predição de intenção a empresa deseja intensificar as propagandas de produtos para que o usuário feche a compra e além disso pode-se criar ofertas personalizadas para que o usuário efetue a compra

**Qual é o processo atual da empresa?** Atualmente a empresa não contém nenhum processo de predizer intenção de compra, o que é feito são ofertas de marketing geral, com esse projeto é possível melhorar o impacto do marketing e gerando maior receita para a empresa

**Qual é o possível potencial de impacto com o projeto?** Com o usuário mais propenso a realizar uma compra, o projeto pode impactar diretamente em transformar a intenção em ação de compra, melhorando as métricas de conversão e aumentando o faturamento da empresa

**Quais são os possíveis riscos?** O usuário pode receber diversas ofertas sem necessidade pois o projeto indicou que ele tem a intenção de comprar gerando um atrito com o usuário perdendo assim um comprador, de outro lado, podemos deixar de ofertar bons produtos ou boas ofertas que levam o usuário a realizar a compra caso o projeto não ter classificado esse usuário corretamente

#### Possíveis soluções

Foi levantado as seguintes possíveis soluções:
- Um produto suportado por machine learning prevendo em tempo real a intenção de compra;	
- Dado uma navegação de usuário o sistema deve auxiliar as áreas interessadas que aquele usuário tem uma intenção de compra e com isso cada uma terá as suas demandas;
- A solução será um modelo supervisionado de classificação;
- Solução deve responder à uma requisição feita por um cliente e a saída deve ser uma probabilidade da intenção de compras em tempo real;

#### Machine learning canvas

OBS: Será realizado após as experimentações e definição do modelo final

### Arquitetura

<p align="center" width="100%">
    <img width="80%" src="https://i.imgur.com/SGriLgE.png">
</p>

### Estrutura do projeto

config: arquivos de configurações
data: dados para o projeto
docs: documentações do projeto
models: arquivos serializados dos modelos utilizandos no projeto
notebooks: notebooks contendo os processos descritos do projeto
src: código fonte do projeto
tests: arquivos de testse dos códigos do projeto

## Insights

Nesta seção será disponibilizados alguns insights retirados da etapa de análise exploratória dos dados, para mais informações acessar o notebook: ```1-eda```.

### Análise Univariada
1. Os dados estão desbalanceados, onde há ~84% para a classe False.
2. Usuários que acessaram o site utilizaram o browser 2 ~65% de todos os acessos

### Análise Bivariada
1. Usuários que compraram acessam quase o dobro de páginas relacionada ao produto do que usuários sem compra
2. Usuários que tiveram a compra realizada possuem taxas de bounce menores do que usuários sem compra, tornando um fator determinante em definição de uma inteção de compra.

### Análise Multivariada
1. Identificamos que quando há uma compra existente o valor da página é exponencialmente maior que uando não existe a conclusão da compra
2. Usuários relacionados aos trafégos: 15, 16, 17, 18 e 19 são usuários que tendem a entrar no site e logo sair e também usuários que realizaram a compra originados do trafégo 14 são usuários que tem uma taxa de bounce maior do que usuários que não tiveram a compra.

## Resultados

TBD

## Como rodar o projeto 

TBD