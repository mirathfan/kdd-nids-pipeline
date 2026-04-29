# =============================================================================
# KDD Cup 1999 - Network Intrusion Detection
# Phase 1: Data Ingestion & Exploratory Data Analysis
# Primary language: R
# =============================================================================

# --- 0. Install & load packages ----------------------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  tidyverse,   # dplyr, ggplot2, readr, tidyr
  ggcorrplot,  # correlation heatmap
  scales,      # axis formatting
  gridExtra,   # multi-plot layout
  knitr        # summary tables
)

# --- 1. Column definitions ---------------------------------------------------
# 41 features from kddcup.names + label column
col_names <- c(
  "duration", "protocol_type", "service", "flag",
  "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
  "hot", "num_failed_logins", "logged_in", "num_compromised",
  "root_shell", "su_attempted", "num_root", "num_file_creations",
  "num_shells", "num_access_files", "num_outbound_cmds",
  "is_host_login", "is_guest_login", "count", "srv_count",
  "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
  "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
  "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
  "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
  "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
  "dst_host_srv_serror_rate", "dst_host_rerror_rate",
  "dst_host_srv_rerror_rate", "label"
)

col_types <- cols(
  duration              = col_double(),
  protocol_type         = col_character(),
  service               = col_character(),
  flag                  = col_character(),
  src_bytes             = col_double(),
  dst_bytes             = col_double(),
  land                  = col_integer(),
  wrong_fragment        = col_double(),
  urgent                = col_double(),
  hot                   = col_double(),
  num_failed_logins     = col_double(),
  logged_in             = col_integer(),
  num_compromised       = col_double(),
  root_shell            = col_double(),
  su_attempted          = col_double(),
  num_root              = col_double(),
  num_file_creations    = col_double(),
  num_shells            = col_double(),
  num_access_files      = col_double(),
  num_outbound_cmds     = col_double(),
  is_host_login         = col_integer(),
  is_guest_login        = col_integer(),
  count                 = col_double(),
  srv_count             = col_double(),
  serror_rate           = col_double(),
  srv_serror_rate       = col_double(),
  rerror_rate           = col_double(),
  srv_rerror_rate       = col_double(),
  same_srv_rate         = col_double(),
  diff_srv_rate         = col_double(),
  srv_diff_host_rate    = col_double(),
  dst_host_count        = col_double(),
  dst_host_srv_count    = col_double(),
  dst_host_same_srv_rate      = col_double(),
  dst_host_diff_srv_rate      = col_double(),
  dst_host_same_src_port_rate = col_double(),
  dst_host_srv_diff_host_rate = col_double(),
  dst_host_serror_rate        = col_double(),
  dst_host_srv_serror_rate    = col_double(),
  dst_host_rerror_rate        = col_double(),
  dst_host_srv_rerror_rate    = col_double(),
  label                       = col_character()
)

# --- 2. Download & load data -------------------------------------------------
data_url  <- "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
local_gz  <- "data/kddcup_10pct.gz"
local_csv <- "data/kddcup_10pct.csv"

dir.create("data",    showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

if (!file.exists(local_gz)) {
  message("Downloading KDD Cup 10% subset (~2 MB compressed)...")
  download.file(data_url, destfile = local_gz, mode = "wb")
}

if (!file.exists(local_csv)) {
  message("Decompressing and parsing data...")
  raw <- read_csv(
    gzcon(file(local_gz, "rb")),
    col_names = col_names,
    col_types = col_types,
    show_col_types = FALSE
  )
  write_csv(raw, local_csv)
  message("Saved to ", local_csv)
} else {
  message("Loading from cache: ", local_csv)
  raw <- read_csv(local_csv, col_types = col_types, show_col_types = FALSE)
}

message(sprintf("Dataset loaded: %s rows x %s columns",
                nrow(raw), ncol(raw)))

# --- 3. Map granular labels -> 5 attack classes ------------------------------
# Source: kddcup.names training_attack_types
dos_attacks   <- c("back","land","neptune","pod","smurf","teardrop",
                   "apache2","udpstorm","processtable","worm")
probe_attacks <- c("ipsweep","nmap","portsweep","satan","mscan","saint")
r2l_attacks   <- c("ftp_write","guess_passwd","imap","multihop","phf",
                   "spy","warezclient","warezmaster","sendmail","named",
                   "snmpgetattack","snmpguess","xlock","xsnoop","httptunnel")
u2r_attacks   <- c("buffer_overflow","loadmodule","perl","rootkit",
                   "ps","sqlattack","xterm")

# Strip trailing period from labels (KDD data quirk)
df <- raw %>%
  mutate(
    label_raw = str_remove(label, "\\.$"),
    attack_class = case_when(
      label_raw == "normal"            ~ "Normal",
      label_raw %in% dos_attacks       ~ "DoS",
      label_raw %in% probe_attacks     ~ "Probe",
      label_raw %in% r2l_attacks       ~ "R2L",
      label_raw %in% u2r_attacks       ~ "U2R",
      TRUE                             ~ "Unknown"
    ),
    attack_class = factor(attack_class,
                          levels = c("Normal","DoS","Probe","R2L","U2R"))
  )

# Sanity check - flag any unmapped labels
unknown_labels <- df %>%
  filter(attack_class == "Unknown") %>%
  distinct(label_raw)

if (nrow(unknown_labels) > 0) {
  warning("Unmapped labels found: ",
          paste(unknown_labels$label_raw, collapse = ", "))
} else {
  message("All labels successfully mapped to 5 attack classes.")
}

# --- 4. Basic dataset overview -----------------------------------------------
message("\n===== DATASET OVERVIEW =====")
message(sprintf("Total records : %s", format(nrow(df), big.mark=",")))
message(sprintf("Features      : %d (41 + label columns)", ncol(df) - 2))
message(sprintf("Missing values: %d", sum(is.na(df))))

class_summary <- df %>%
  count(attack_class) %>%
  mutate(
    pct        = n / sum(n) * 100,
    pct_label  = sprintf("%.2f%%", pct)
  )

message("\nClass distribution:")
print(class_summary)

# --- 5. EDA Plot 1: Class distribution (log scale) ---------------------------
p1 <- ggplot(class_summary, aes(x = reorder(attack_class, -n),
                                 y = n, fill = attack_class)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = paste0(pct_label, "\n(", format(n, big.mark=","), ")")),
            vjust = -0.4, size = 3, color = "gray30") +
  scale_y_log10(labels = label_comma()) +
  scale_fill_manual(values = c(
    "Normal" = "#378ADD",
    "DoS"    = "#E24B4A",
    "Probe"  = "#EF9F27",
    "R2L"    = "#1D9E75",
    "U2R"    = "#7F77DD"
  )) +
  labs(
    title    = "Class distribution (log scale)",
    subtitle = "KDD Cup 1999 — 10% subset",
    x        = "Attack class",
    y        = "Count (log scale)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray50"),
    panel.grid.major.x = element_blank()
  )

# --- 6. EDA Plot 2: Protocol type by class -----------------------------------
p2 <- df %>%
  count(protocol_type, attack_class) %>%
  group_by(attack_class) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = attack_class, y = prop, fill = protocol_type)) +
  geom_col(width = 0.65) +
  scale_y_continuous(labels = label_percent()) +
  scale_fill_manual(values = c("tcp"="#378ADD","udp"="#EF9F27","icmp"="#E24B4A")) +
  labs(
    title = "Protocol type breakdown by attack class",
    x = "Attack class", y = "Proportion", fill = "Protocol"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.major.x = element_blank()
  )

# --- 7. EDA Plot 3: Key numeric feature distributions by class ---------------
key_features <- c("duration","src_bytes","dst_bytes","count","serror_rate","same_srv_rate")

df_long <- df %>%
  select(attack_class, all_of(key_features)) %>%
  pivot_longer(-attack_class, names_to = "feature", values_to = "value") %>%
  mutate(value_log = log1p(value))   # log1p to handle zeros

p3 <- ggplot(df_long, aes(x = attack_class, y = value_log, fill = attack_class)) +
  geom_boxplot(outlier.size = 0.3, outlier.alpha = 0.2, show.legend = FALSE) +
  facet_wrap(~feature, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c(
    "Normal" = "#378ADD", "DoS" = "#E24B4A",
    "Probe"  = "#EF9F27", "R2L" = "#1D9E75", "U2R" = "#7F77DD"
  )) +
  labs(
    title    = "Feature distributions by attack class (log1p scale)",
    subtitle = "Key numeric features",
    x        = NULL, y = "log1p(value)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title  = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 8),
    strip.text  = element_text(face = "bold")
  )

# --- 8. EDA Plot 4: Correlation heatmap (numeric features only) --------------
numeric_df <- df %>%
  select(where(is.numeric), -matches("land|logged_in|is_host|is_guest")) %>%
  select(-any_of(c("label_raw")))

cor_matrix <- cor(numeric_df, use = "pairwise.complete.obs")

p4 <- ggcorrplot(
  cor_matrix,
  type        = "lower",
  lab         = FALSE,
  tl.cex      = 6,
  colors      = c("#E24B4A", "white", "#378ADD"),
  title       = "Feature correlation matrix",
  ggtheme     = theme_minimal(base_size = 10)
) +
  theme(plot.title = element_text(face = "bold", size = 14))

# --- 9. EDA Plot 5: Top 10 raw attack labels ---------------------------------
p5 <- df %>%
  count(label_raw, attack_class, sort = TRUE) %>%
  slice_head(n = 15) %>%
  ggplot(aes(x = reorder(label_raw, n), y = n, fill = attack_class)) +
  geom_col(width = 0.7, show.legend = TRUE) +
  coord_flip() +
  scale_y_continuous(labels = label_comma()) +
  scale_fill_manual(values = c(
    "Normal" = "#378ADD", "DoS" = "#E24B4A",
    "Probe"  = "#EF9F27", "R2L" = "#1D9E75", "U2R" = "#7F77DD"
  )) +
  labs(
    title = "Top 15 raw attack subtypes",
    x     = NULL, y = "Count", fill = "Class"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.major.y = element_blank()
  )

# --- 10. Save all plots -------------------------------------------------------
message("\nSaving plots to outputs/...")

ggsave("outputs/p1_class_distribution.png", p1, width=8, height=5, dpi=150)
ggsave("outputs/p2_protocol_by_class.png",  p2, width=8, height=5, dpi=150)
ggsave("outputs/p3_feature_distributions.png", p3, width=10, height=7, dpi=150)
ggsave("outputs/p4_correlation_heatmap.png",   p4, width=9,  height=8, dpi=150)
ggsave("outputs/p5_attack_subtypes.png",    p5, width=8, height=6, dpi=150)

message("All plots saved.")

# --- 11. Export clean labelled data for Phase 2 ------------------------------
df_clean <- df %>%
  select(-label, -label_raw) %>%          # drop raw label columns
  rename(label = attack_class)            # use 5-class label going forward

write_csv(df_clean, "data/kdd_labelled.csv")
message(sprintf("\nPhase 1 complete. Clean dataset saved: data/kdd_labelled.csv (%s rows)",
                format(nrow(df_clean), big.mark=",")))

# --- 12. Quick summary table -------------------------------------------------
cat("\n===== FINAL CLASS SUMMARY =====\n")
df_clean %>%
  count(label) %>%
  mutate(
    percent    = sprintf("%.3f%%", n / sum(n) * 100),
    imbalanced = ifelse(n / sum(n) < 0.01, "YES - needs SMOTE", "ok")
  ) %>%
  knitr::kable(col.names = c("Class","Count","% of total","Imbalanced?"),
               format = "simple") %>%
  print()
