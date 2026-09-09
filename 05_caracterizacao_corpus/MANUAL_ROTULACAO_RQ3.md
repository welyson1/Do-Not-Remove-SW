# Manual de Rotulacao da RQ3

Este manual descreve o fluxo atual de rotulacao dos comentarios da RQ3 com 3 rotuladores fixos.

## Arquivos de entrada para o software

Use somente estes CSVs como entrada no software:

- `11_amostra_piloto_concordancia_300_comentarios.csv`: rodada piloto, com 300 comentarios no total e 50 por classe.
- `10_amostra_rotulacao_completa_4800_comentarios.csv`: rodada principal, com 4.800 comentarios no total e 800 por classe.

Os dois arquivos mantem o mesmo cabecalho do CSV antigo para nao quebrar o software:

```text
sample_id,sample_class,sample_class_label,sample_year,comment_id,video_id,published_at,is_reply,like_count,comment_length,label_combo,n_labels,C1,C2,C3,C4,C5,C6,comment
```

## Arquivo baixado do software

Ao final da rotulacao, o software deve exportar um CSV no formato:

```text
comment_id,video_id,comment,C1,C2,C3,C4,C5,C6
```

Os valores de `C1` a `C6` devem ser `0` ou `1`. Celulas vazias sao aceitas pelo script, mas a respectiva classe ficara marcada para revisao se nao houver maioria.

## Fluxo recomendado

1. Carregue `11_amostra_piloto_concordancia_300_comentarios.csv` no software.
2. Os 3 rotuladores exportam seus arquivos do piloto, por exemplo:
   - `rotulos-piloto-Welyson.csv`
   - `rotulos-piloto-Avaliador2.csv`
   - `rotulos-piloto-Avaliador3.csv`
3. Calcule a concordancia preliminar do piloto:

```powershell
python 05_caracterizacao_corpus\etapa_14_unificar_rotulos_rq3.py --pilot-labels rotulos-piloto-Welyson.csv rotulos-piloto-Avaliador2.csv rotulos-piloto-Avaliador3.csv
```

4. Abra `05_caracterizacao_corpus/13_relatorio_unificacao_rotulos_rq3.txt` e confira:
   - se os 3 arquivos pertencem a amostra piloto;
   - alfa de Krippendorff por classe;
   - Kappa de Fleiss por classe;
   - quantas linhas ficaram marcadas para revisao.
5. Ajuste o protocolo de rotulacao se necessario.
6. Carregue `10_amostra_rotulacao_completa_4800_comentarios.csv` no software para os mesmos 3 rotuladores.
7. Os 3 rotuladores exportam seus arquivos completos, por exemplo:
   - `rotulos-Welyson.csv`
   - `rotulos-Avaliador2.csv`
   - `rotulos-Avaliador3.csv`
8. Gere o relatorio final de concordancia e o CSV final:

```powershell
python 05_caracterizacao_corpus\etapa_14_unificar_rotulos_rq3.py --full-labels rotulos-Welyson.csv rotulos-Avaliador2.csv rotulos-Avaliador3.csv
```

Tambem e possivel calcular piloto e final no mesmo comando:

```powershell
python 05_caracterizacao_corpus\etapa_14_unificar_rotulos_rq3.py --pilot-labels rotulos-piloto-Welyson.csv rotulos-piloto-Avaliador2.csv rotulos-piloto-Avaliador3.csv --full-labels rotulos-Welyson.csv rotulos-Avaliador2.csv rotulos-Avaliador3.csv
```

## Saidas geradas

Depois da rodada piloto:

- `13_relatorio_unificacao_rotulos_rq3.txt`
- `13_rotulos_piloto_concordancia_consolidado.csv`

Depois da rodada completa:

- `13_relatorio_unificacao_rotulos_rq3.txt`
- `14_rotulos_rq3_final_4800.csv`

O arquivo final contem os 4.800 comentarios, ordenados pela amostra completa, com os rotulos finais `C1` a `C6` definidos por maioria entre os 3 rotuladores.

## Regras usadas pelo script

- A amostra piloto tem 300 comentarios: 50 por classe.
- A amostra completa tem 4.800 comentarios: 800 por classe.
- O piloto e subconjunto da amostra completa.
- Os mesmos 3 rotuladores devem rotular todos os 4.800 comentarios.
- O rotulo final e definido por maioria simples entre os 3 rotuladores.
- Se nao houver maioria ou houver anotacoes ausentes demais, a linha fica marcada para revisao.
- Comentarios principais e replies sao tratados como comentarios.

## Alertas importantes

- Nao use arquivos de rotulos gerados a partir da amostra antiga de 4.000 comentarios.
- O `rotulos-Welyson.csv` antigo pertence a amostra antiga; substitua-o pelo novo arquivo exportado apos rotular os 4.800 comentarios.
- Se um arquivo de rotulos nao corresponder a amostra esperada, o script interrompe a execucao e informa quantos IDs estao fora ou ausentes.

## Regenerar as amostras

Se for necessario recriar os CSVs de entrada:

```powershell
python 05_caracterizacao_corpus\etapa_13_amostrar_comentarios_rq3.py
```

Depois disso, confira novamente o relatorio:

```powershell
python 05_caracterizacao_corpus\etapa_14_unificar_rotulos_rq3.py
```
