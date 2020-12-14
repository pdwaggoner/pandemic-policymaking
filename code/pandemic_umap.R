# Replication code for pandemic policymaking paper (https://arxiv.org/abs/2011.04763)
# Philip Waggoner (pdwaggoner@uchicago.edu)

# libs
library(tidyverse)
library(recipes)
library(skimr)
library(lubridate)
library(umap)

# start with some basic feature engineering
bills <- bills %>%
  mutate(Committees = as.factor(Committees),
         Senate = as.factor(ifelse(Type_Chamber == "S", 1, 0)),
         Type_Bill = as.factor(case_when(
           Type_Bill == "B" ~ 1,
           Type_Bill == "Con" ~ 2,
           Type_Bill == "J" ~ 3,
           Type_Bill == "Res" ~ 4)),
         Party = as.factor(case_when(
           Party == "D" ~ 1,
           Party == "I" ~ 2,
           Party == "ID" ~ 2,
           Party == "R" ~ 3)),
         State_Factor = as.factor(State),
         Date_Intro = mdy(Date_Intro),
         Date_Last = mdy(Date_Last),
         Year = Date
         ) %>%
  select(-Type_Chamber, -Type_Number, -Title, -Chamber,
         -Sponsor, -District, -Last_Action, -Date,
         -State) %>%
  relocate(Party)

recipe <- recipe(Senate ~ .,
                 data = bills) %>%
  step_knnimpute(all_predictors())

bills_imputed <- prep(recipe) %>%
    juice()

bills_imputed <- bills_imputed %>%
  mutate(Type_Bill = as.numeric(Type_Bill),
         Committees = as.numeric(Committees),
         State_Factor = as.numeric(State_Factor),
         Senate = as.numeric(Senate),
         Date_Intro = as.numeric(Date_Intro),
         Date_Last = as.numeric(Date_Last))

library(tictoc)

{
  hyperparameters_umap <- expand.grid(n_neighbors = seq(5, 50, 10),
                                      n_epochs    = seq(50, 450, 100))
  tic()
  umap_full <- pmap(hyperparameters_umap,
                    umap,
                    d = bills_imputed[,2:10])
  toc()
  } # ~1.9 min

grid_values_umap <- tibble(n_neighbors = rep(hyperparameters_umap$n_neighbors, each = 1507),
                           n_epochs = rep(hyperparameters_umap$n_epochs, each = 1507),
                           d1 = unlist(map(umap_full, ~ .$layout[, 1])),
                           d2 = unlist(map(umap_full, ~ .$layout[, 2])))

grid_values_umap %>%
  ggplot(aes(d1, d2)) +
  geom_point() +
  facet_grid(n_neighbors ~ n_epochs,
             scales = "fixed") +
  labs(title = "Hyperparameter Grid Search",
       subtitle = "Uniform Manifold Approximation and Projection",
       x = "First Dimension",
       y = "Second Dimension",
       caption = "Columns are Epochs\nRows are Neighborhood Sizes") +
  theme_minimal()

# COVID period
bills_imputed_covid <- bills_imputed %>%
  filter(Year >= 2019)

{
  suppressWarnings(
    umap_covid <- bills_imputed_covid[ , 2:10] %>%
      umap(n_neighbors = 45,
           metric = "euclidean",
           n_epochs = 450)
  )

  suppressWarnings(
    umap_covid <- bills_imputed_covid %>%
      mutate_if(.funs = scale,
                .predicate = is.numeric,
                scale = FALSE) %>%
      mutate(First_Dimension = umap_covid$layout[,1],
             Second_Dimension = umap_covid$layout[,2]) %>%
      gather(key = "Variable",
             value = "Value",
             c(-First_Dimension, -Second_Dimension, -Party))
  )
}


library(amerika)

covid <- ggplot(umap_covid, aes(First_Dimension, Second_Dimension,
                              col = factor(Party))) +
  geom_point() +
  scale_color_manual(values=c(amerika_palettes$Dem_Ind_Rep3[1],
                              amerika_palettes$Dem_Ind_Rep3[2],
                              amerika_palettes$Dem_Ind_Rep3[3]),
                     name="Party",
                     breaks=c("1",
                              "2",
                              "3"),
                     labels=c("Dem",
                              "Ind",
                              "Rep")) +
  labs(title = "COVID",
       subtitle = "Neighborhood size: 45; Epochs = 450",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()
covid

# Pre-COVID period
bills_imputed_pre <- bills_imputed %>%
  filter(Year < 2019)

{
  suppressWarnings(
    umap_pre <- bills_imputed_pre[ , 2:10] %>%
      umap(n_neighbors = 45,
           metric = "euclidean",
           n_epochs = 450)
  )

  suppressWarnings(
    umap_pre <- bills_imputed_pre %>%
      mutate_if(.funs = scale,
                .predicate = is.numeric,
                scale = FALSE) %>%
      mutate(First_Dimension = umap_pre$layout[,1],
             Second_Dimension = umap_pre$layout[,2]) %>%
      gather(key = "Variable",
             value = "Value",
             c(-First_Dimension, -Second_Dimension, -Party))
  )
}

pre_covid <- ggplot(umap_pre, aes(First_Dimension, Second_Dimension,
                              col = factor(Party))) +
  geom_point() +
  scale_color_manual(values=c(amerika_palettes$Dem_Ind_Rep3[1],
                              amerika_palettes$Dem_Ind_Rep3[2],
                              amerika_palettes$Dem_Ind_Rep3[3]),
                     name="Party",
                     breaks=c("1",
                              "2",
                              "3"),
                     labels=c("Dem",
                              "Ind",
                              "Rep")) +
  labs(title = "Pre-COVID",
       subtitle = "Neighborhood size: 45; Epochs = 450",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()
pre_covid

# supervised projection
umap_plot <- function(x, labels,
                   main="Projecting COVID onto Pre-COVID Embedding (Excluding Time-Related Features)",
                   xlab = "First Dimension",
                   ylab = "Second Dimension",
                   colors=c(
                     amerika_palettes$Dem_Ind_Rep3[1],
                     amerika_palettes$Dem_Ind_Rep3[2],
                     amerika_palettes$Dem_Ind_Rep3[3]
                   ),
                   pad=0.1, cex=0.65, pch=19,
                   add=FALSE, legend.suffix="",
                   cex.main=1, cex.legend=1) {

            layout = x
            if (is(x, "umap")) {
              layout = x$layout
            }

            xylim = range(layout)
            xylim = xylim + ((xylim[2]-xylim[1])*pad)*c(-0.5, 0.5)
            if (!add) {
              par(mar=c(0.2,0.7,1.2,0.7), ps=10)
              plot(xylim, xylim,
                   type="n",
                   axes=F, frame=F)
            }
            points(layout[,1], layout[,2], col=c(
              amerika_palettes$Dem_Ind_Rep3[1],
              amerika_palettes$Dem_Ind_Rep3[2],
              amerika_palettes$Dem_Ind_Rep3[3]
            ),
                   cex=cex, pch=pch)
            mtext(side=3, main, cex=cex.main)

            labels.u = unique(labels)
            legend.pos = "topright"
            legend.text = as.character(labels.u)
            if (add) {
              legend.pos = "bottomright"
              legend.text = paste(as.character(labels.u), legend.suffix)
            }
            legend(legend.pos, legend=legend.text,
                   col=c(
                     amerika_palettes$Dem_Ind_Rep3[1],
                     amerika_palettes$Dem_Ind_Rep3[2],
                     amerika_palettes$Dem_Ind_Rep3[3]
                   ),
                   bty="n", pch=c("D", "I", "R"), cex=cex.legend)
          }

pre_covid_data <- bills_imputed_pre[, 2:10]
pre_covid_labels <- bills_imputed_pre$Party

pre_covid_umap <- umap(pre_covid_data)

covid_data <- bills_imputed_covid[, 2:10]
colnames(covid_data) <- colnames(covid_data)
covid_preds <- predict(pre_covid_umap, covid_data)

{
umap_plot(pre_covid_umap, pre_covid_labels)
umap_plot(covid_preds, pre_covid_labels, add=TRUE, pch=4,
          legend.suffix=" (X = covid)",
          cex = 1.5)
grid(lty = "solid")
rect(-13.5, 4.5, -11.75, 6.5,
     border = "green")
}

# (now, without time features)
pre_covid_data <- bills_imputed_pre[, c("Type_Bill", "Cosponsors", "Committees",
                                        "State_Factor", "Senate")]
pre_covid_labels <- bills_imputed_pre$Party

pre_covid_umap <- umap(pre_covid_data)

covid_data <- bills_imputed_covid[, c("Type_Bill", "Cosponsors", "Committees",
                                      "State_Factor", "Senate")]
colnames(covid_data) <- colnames(covid_data)
covid_preds <- predict(pre_covid_umap, covid_data)

{
  umap_plot(pre_covid_umap, pre_covid_labels)
  umap_plot(covid_preds, pre_covid_labels, add=TRUE, pch=4,
            legend.suffix=" (X = covid)",
            cex = 1.5)
  grid(lty = "solid")
}

# new task via tidymodels
library(tidymodels)

set.seed(1234)

split_tidy <- initial_split(bills_imputed,
                            prop = 0.664) # same proportion as COVID (379) to pre-COVID (1128) bills -> 0.33

train_tidy <- training(split_tidy)
test_tidy  <- testing(split_tidy)

umap_plot <- function(x, labels,
                      main="Projecting Random Test Set onto Training Set Embedding",
                      xlab = "First Dimension",
                      ylab = "Second Dimension",
                      colors=c(
                        amerika_palettes$Dem_Ind_Rep3[1],
                        amerika_palettes$Dem_Ind_Rep3[2],
                        amerika_palettes$Dem_Ind_Rep3[3]
                      ),
                      pad=0.1, cex=0.65, pch=19,
                      add=FALSE, legend.suffix="",
                      cex.main=1, cex.legend=1) {

  layout = x
  if (is(x, "umap")) {
    layout = x$layout
  }

  xylim = range(layout)
  xylim = xylim + ((xylim[2]-xylim[1])*pad)*c(-0.5, 0.5)
  if (!add) {
    par(mar=c(0.2,0.7,1.2,0.7), ps=10)
    plot(xylim, xylim,
         type="n",
         axes=F, frame=F)
  }
  points(layout[,1], layout[,2], col=c(
    amerika_palettes$Dem_Ind_Rep3[1],
    amerika_palettes$Dem_Ind_Rep3[2],
    amerika_palettes$Dem_Ind_Rep3[3]
  ),
  cex=cex, pch=pch)
  mtext(side=3, main, cex=cex.main)

  labels.u = unique(labels)
  legend.pos = "topright"
  legend.text = as.character(labels.u)
  if (add) {
    legend.pos = "bottomright"
    legend.text = paste(as.character(labels.u), legend.suffix)
  }
  legend(legend.pos, legend=legend.text,
         col=c(
           amerika_palettes$Dem_Ind_Rep3[1],
           amerika_palettes$Dem_Ind_Rep3[2],
           amerika_palettes$Dem_Ind_Rep3[3]
         ),
         bty="n", pch=c("D", "I", "R"), cex=cex.legend)
}

pre_covid_data <- train_tidy[, c("Type_Bill", "Cosponsors", "Committees",
                                 "State_Factor", "Senate")]
pre_covid_labels <- train_tidy$Party

pre_covid_umap <- umap(pre_covid_data)

covid_data <- test_tidy[, c("Type_Bill", "Cosponsors", "Committees",
                            "State_Factor", "Senate")]
colnames(covid_data) <- colnames(covid_data)
covid_preds <- predict(pre_covid_umap, covid_data)

{
  umap_plot(pre_covid_umap, pre_covid_labels)
  umap_plot(covid_preds, pre_covid_labels, add=TRUE, pch=4,
            legend.suffix=" (X = test set)",
            cex = 1.5)
  grid(lty = "solid")
}
