# Protocolo de Auditoria da RQ3

## Objetivo

Verificar se cada comentario selecionado realmente corresponde a classe prevista pelo dicionario lexical. Esta auditoria estima a precisao das classes, nao reclassifica o corpus inteiro.

## Rodada concluida

- Arquivo auditado: `05_caracterizacao_corpus/06_auditoria_precisao_classes.csv`.
- Amostra: 300 comentarios, com 50 por classe, sem repeticao de comentario normalizado.
- Resultado: 277 classificacoes confirmadas como corretas e 23 incorretas.
- Macro-precisao: 92,33%.
- Precisao por classe: C1 = 96,00%; C2 = 92,00%; C3 = 96,00%; C4 = 88,00%; C5 = 92,00%; C6 = 90,00%.
- Relatorio: `05_caracterizacao_corpus/07_relatorio_precisao_classes.txt`.

## Como decidir

Para cada linha, leia a classe avaliada, as regras acionadas e o comentario completo.

Marque `Sim, pertence a classe prevista` quando o comentario tiver a funcao discursiva da classe mostrada. Em classificacao multirrotulo, isso continua sendo `Sim` mesmo que o comentario tambem pertenca a outra classe.

Marque `Nao, nao pertence a classe prevista` quando a funcao discursiva da classe mostrada nao aparecer no comentario. Nesse caso, selecione uma ou mais alternativas em `Classes alternativas`.

Deixe como `Ainda nao avaliei` apenas quando quiser pular temporariamente a linha.

## Criterio por classe

- `C1`: pergunta, pedido de ajuda, pedido de explicacao, orientacao, exemplo ou tutorial.
- `C2`: problema operacional, erro, falha, bloqueio, mau funcionamento ou dificuldade concreta de uso.
- `C3`: elogio, agradecimento, testemunho positivo ou reconhecimento de utilidade.
- `C4`: critica, frustracao, objecao, custo percebido como ruim, limitacao ou avaliacao negativa.
- `C5`: recomendacao, divulgacao ou encaminhamento a recursos/alternativas, incluindo convite, link, canal, curso, playlist, documentacao, forum, comunidade ou alternativa sugerida.
- `C6`: humor, ruido evidente, informalidade dominante, link isolado ou comentario pouco analitico.

## Casos ambiguos

Se o comentario tiver a classe avaliada e outra classe ao mesmo tempo, marque `Sim`.

Se a regra disparou por uma palavra solta, mas o sentido do comentario nao corresponde a classe, marque `Nao`.

Se mais de uma classe descrever melhor o comentario, selecione todas elas em `Classes alternativas`.

Se nao houver conteudo suficiente para decidir, marque `Nao` e escolha `sem_classe`.

Use `Observacoes opcionais` apenas para justificar casos dificeis, falso positivo evidente ou ambiguidade relevante.

## Depois de auditar

Quando terminar as 300 linhas, clique em `Atualizar relatorio de precisao` no app ou execute:

```powershell
python 05_caracterizacao_corpus/etapa_12_calcular_precisao_classes.py
```

O resultado sera atualizado em `05_caracterizacao_corpus/07_relatorio_precisao_classes.txt`.

## Se aparecer erro visual no navegador

Se aparecer `removeChild` ou erro parecido do Streamlit, recarregue a pagina com `Ctrl+F5`. A aplicacao foi ajustada para usar componentes nativos e texto simples no comentario, o que reduz bastante esse problema ao usar traducao automatica. Se ainda ocorrer durante o clique em salvar, salve sem traducao ativa e reative a traducao apenas para leitura dos comentarios.
