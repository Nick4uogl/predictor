import { useState, useEffect, FormEvent } from "react";
import {
  Typography,
  Box,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Stack,
  TextField,
  Checkbox,
  FormControlLabel,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import {
  Delete as DeleteIcon,
  Edit as EditIcon,
  Notifications as NotificationsIcon,
} from "@mui/icons-material";
import {
  NotificationSettings,
  NotificationHistory,
  api,
} from "../services/api";
import { SelectChangeEvent } from "@mui/material/Select";

// Компонент сторінки сповіщень
const Notifications = () => {
  // Електронна пошта користувача
  const [email, setEmail] = useState<string>("");
  const [emailInput, setEmailInput] = useState<string>("");

  // Стан для форми створення/редагування сповіщення
  const [cities, setCities] = useState<string[]>([]);
  const [selectedCity, setSelectedCity] = useState<string>("");
  const [showForm, setShowForm] = useState<boolean>(false);
  const [editingNotification, setEditingNotification] =
    useState<NotificationSettings | null>(null);

  // Порогові значення
  const [thresholdPM25, setThresholdPM25] = useState<string>("");
  const [thresholdPM10, setThresholdPM10] = useState<string>("");
  const [thresholdO3, setThresholdO3] = useState<string>("");
  const [thresholdNO2, setThresholdNO2] = useState<string>("");
  const [thresholdSO2, setThresholdSO2] = useState<string>("");
  const [thresholdCO, setThresholdCO] = useState<string>("");
  const [thresholdAQI, setThresholdAQI] = useState<string>("");
  const [dailySummary, setDailySummary] = useState<boolean>(false);

  // Стан для списку сповіщень та їх історії
  const [notifications, setNotifications] = useState<NotificationSettings[]>(
    []
  );
  const [history, setHistory] = useState<NotificationHistory[]>([]);

  // Стани завантаження і помилок
  const [loading, setLoading] = useState<boolean>(false);
  const [fetchingCities, setFetchingCities] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [success, setSuccess] = useState<string>("");

  // Стан для діалогу підтвердження видалення
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<boolean>(false);
  const [notificationToDelete, setNotificationToDelete] = useState<
    number | null
  >(null);

  // Отримати список міст при завантаженні
  useEffect(() => {
    const fetchCities = async () => {
      setFetchingCities(true);
      try {
        const citiesData = await api.getCities();
        setCities(citiesData);
      } catch (error) {
        console.error("Error fetching cities:", error);
        setError("Не вдалося отримати список міст. Спробуйте пізніше.");
      } finally {
        setFetchingCities(false);
      }
    };

    fetchCities();
  }, []);

  // Функція для отримання сповіщень для електронної пошти
  const handleEmailSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!emailInput) {
      setError("Будь ласка, вкажіть електронну пошту");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const notificationsData = await api.getNotifications(emailInput);
      setNotifications(notificationsData);

      const historyData = await api.getNotificationHistory(emailInput);
      setHistory(historyData);

      setEmail(emailInput);
    } catch (error) {
      console.error("Error fetching notifications:", error);
      setError(
        "Не вдалося отримати сповіщення. Перевірте електронну пошту та спробуйте знову."
      );
    } finally {
      setLoading(false);
    }
  };

  // Функція для відкриття форми створення нового сповіщення
  const handleNewNotification = () => {
    // Скинути всі поля форми
    setSelectedCity("");
    setThresholdPM25("");
    setThresholdPM10("");
    setThresholdO3("");
    setThresholdNO2("");
    setThresholdSO2("");
    setThresholdCO("");
    setThresholdAQI("");
    setDailySummary(false);

    setEditingNotification(null);
    setShowForm(true);
  };

  // Функція для відкриття форми редагування існуючого сповіщення
  const handleEditNotification = (notification: NotificationSettings) => {
    setSelectedCity(notification.city_name);
    setThresholdPM25(notification.threshold_pm25?.toString() || "");
    setThresholdPM10(notification.threshold_pm10?.toString() || "");
    setThresholdO3(notification.threshold_o3?.toString() || "");
    setThresholdNO2(notification.threshold_no2?.toString() || "");
    setThresholdSO2(notification.threshold_so2?.toString() || "");
    setThresholdCO(notification.threshold_co?.toString() || "");
    setThresholdAQI(notification.threshold_aqi?.toString() || "");
    setDailySummary(notification.daily_summary);

    setEditingNotification(notification);
    setShowForm(true);
  };

  // Функція для зміни активного статусу сповіщення
  const handleToggleActive = async (notification: NotificationSettings) => {
    if (!notification.id) return;

    setLoading(true);
    setError("");
    try {
      await api.updateNotification(notification.id, {
        is_active: !notification.is_active,
      });

      // Оновити список сповіщень
      const notificationsData = await api.getNotifications(email);
      setNotifications(notificationsData);

      setSuccess(
        `Статус сповіщення для міста ${notification.city_name} змінено на ${
          !notification.is_active ? "активний" : "неактивний"
        }`
      );

      // Скинути повідомлення про успіх через 3 секунди
      setTimeout(() => {
        setSuccess("");
      }, 3000);
    } catch (error) {
      console.error("Error updating notification:", error);
      setError("Не вдалося змінити статус сповіщення. Спробуйте пізніше.");
    } finally {
      setLoading(false);
    }
  };

  // Функція для виклику діалогу підтвердження видалення
  const handleConfirmDelete = (id: number | undefined) => {
    if (!id) return;
    setNotificationToDelete(id);
    setDeleteDialogOpen(true);
  };

  // Функція для закриття діалогу без видалення
  const handleCancelDelete = () => {
    setDeleteDialogOpen(false);
    setNotificationToDelete(null);
  };

  // Функція для видалення сповіщення
  const handleDeleteNotification = async () => {
    if (!notificationToDelete) return;

    setLoading(true);
    setError("");
    try {
      await api.deleteNotification(notificationToDelete);

      // Оновити список сповіщень
      const notificationsData = await api.getNotifications(email);
      setNotifications(notificationsData);

      setSuccess("Сповіщення успішно видалено");

      // Скинути повідомлення про успіх через 3 секунди
      setTimeout(() => {
        setSuccess("");
      }, 3000);
    } catch (error) {
      console.error("Error deleting notification:", error);
      setError("Не вдалося видалити сповіщення. Спробуйте пізніше.");
    } finally {
      setLoading(false);
      setDeleteDialogOpen(false);
      setNotificationToDelete(null);
    }
  };

  // Обробник зміни міста
  const handleCityChange = (event: SelectChangeEvent) => {
    setSelectedCity(event.target.value);
  };

  // Функція для відправки форми створення/редагування сповіщення
  const handleSubmitForm = async (e: FormEvent) => {
    e.preventDefault();

    if (!selectedCity) {
      setError("Будь ласка, виберіть місто");
      return;
    }

    // Підготувати дані для сповіщення
    const notificationData: NotificationSettings = {
      email: email,
      city_name: selectedCity,
      threshold_pm25: thresholdPM25 ? parseFloat(thresholdPM25) : null,
      threshold_pm10: thresholdPM10 ? parseFloat(thresholdPM10) : null,
      threshold_o3: thresholdO3 ? parseFloat(thresholdO3) : null,
      threshold_no2: thresholdNO2 ? parseFloat(thresholdNO2) : null,
      threshold_so2: thresholdSO2 ? parseFloat(thresholdSO2) : null,
      threshold_co: thresholdCO ? parseFloat(thresholdCO) : null,
      threshold_aqi: thresholdAQI ? parseFloat(thresholdAQI) : null,
      daily_summary: dailySummary,
      is_active: true,
    };

    setLoading(true);
    setError("");
    try {
      if (editingNotification && editingNotification.id) {
        // Редагування існуючого сповіщення
        await api.updateNotification(editingNotification.id, notificationData);
        setSuccess(`Сповіщення для міста ${selectedCity} успішно оновлено`);
      } else {
        // Створення нового сповіщення
        await api.createNotification(notificationData);
        setSuccess(`Сповіщення для міста ${selectedCity} успішно створено`);
      }

      // Оновити список сповіщень
      const notificationsData = await api.getNotifications(email);
      setNotifications(notificationsData);

      // Скинути форму
      setShowForm(false);

      // Скинути повідомлення про успіх через 3 секунди
      setTimeout(() => {
        setSuccess("");
      }, 3000);
    } catch (error) {
      console.error("Error saving notification:", error);
      setError("Не вдалося зберегти сповіщення. Спробуйте пізніше.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        width: "100%",
        maxWidth: "100%",
        pt: 2,
        pb: 4,
        px: { xs: 2, sm: 3 },
        boxSizing: "border-box",
      }}
    >
      <Typography variant="h4" component="h1" gutterBottom>
        Сповіщення про якість повітря
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2, width: "100%" }}>
          {error}
        </Alert>
      )}

      {success && (
        <Alert severity="success" sx={{ mb: 2, width: "100%" }}>
          {success}
        </Alert>
      )}

      <Paper elevation={3} sx={{ p: 4, mb: 4, width: "100%" }}>
        <Typography variant="h5" gutterBottom>
          Введіть вашу електронну пошту
        </Typography>

        <form onSubmit={handleEmailSubmit}>
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems="center"
          >
            <TextField
              label="Електронна пошта"
              value={emailInput}
              onChange={(e) => setEmailInput(e.target.value)}
              fullWidth
              required
              type="email"
            />
            <Button
              type="submit"
              variant="contained"
              disabled={loading}
              sx={{ width: { xs: "100%", md: "auto" } }}
            >
              {loading ? (
                <Box sx={{ display: "flex", alignItems: "center" }}>
                  <CircularProgress size={20} sx={{ mr: 1 }} color="inherit" />
                  Завантаження...
                </Box>
              ) : (
                "Отримати сповіщення"
              )}
            </Button>
          </Stack>
        </form>
      </Paper>

      {email && (
        <>
          <Paper elevation={3} sx={{ p: 4, mb: 4, width: "100%" }}>
            <Stack
              direction="row"
              justifyContent="space-between"
              alignItems="center"
              sx={{ mb: 2 }}
            >
              <Typography variant="h5" gutterBottom>
                Ваші сповіщення
              </Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<NotificationsIcon />}
                onClick={handleNewNotification}
              >
                Додати сповіщення
              </Button>
            </Stack>

            {notifications.length === 0 ? (
              <Alert severity="info">
                У вас ще немає налаштованих сповіщень. Додайте перше сповіщення,
                щоб отримувати інформацію про якість повітря.
              </Alert>
            ) : (
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Місто</TableCell>
                      <TableCell>Тип сповіщення</TableCell>
                      <TableCell>Статус</TableCell>
                      <TableCell>Дії</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {notifications.map((notification) => (
                      <TableRow key={notification.id}>
                        <TableCell>{notification.city_name}</TableCell>
                        <TableCell>
                          {notification.daily_summary && (
                            <Chip
                              label="Щоденний звіт"
                              color="primary"
                              size="small"
                              sx={{ mr: 1, mb: 1 }}
                            />
                          )}
                          {notification.threshold_aqi && (
                            <Chip
                              label={`AQI > ${notification.threshold_aqi}`}
                              color="secondary"
                              size="small"
                              sx={{ mr: 1, mb: 1 }}
                            />
                          )}
                          {notification.threshold_pm25 && (
                            <Chip
                              label={`PM2.5 > ${notification.threshold_pm25}`}
                              color="warning"
                              size="small"
                              sx={{ mr: 1, mb: 1 }}
                            />
                          )}
                          {notification.threshold_pm10 && (
                            <Chip
                              label={`PM10 > ${notification.threshold_pm10}`}
                              color="warning"
                              size="small"
                              sx={{ mr: 1, mb: 1 }}
                            />
                          )}
                          {/* Інші пороги можна додати при потребі */}
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={
                              notification.is_active ? "Активне" : "Неактивне"
                            }
                            color={
                              notification.is_active ? "success" : "default"
                            }
                            onClick={() => handleToggleActive(notification)}
                          />
                        </TableCell>
                        <TableCell>
                          <IconButton
                            onClick={() => handleEditNotification(notification)}
                          >
                            <EditIcon />
                          </IconButton>
                          <IconButton
                            onClick={() => handleConfirmDelete(notification.id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Paper>

          {showForm && (
            <Paper elevation={3} sx={{ p: 4, mb: 4, width: "100%" }}>
              <Typography variant="h5" gutterBottom>
                {editingNotification
                  ? "Редагувати сповіщення"
                  : "Нове сповіщення"}
              </Typography>

              <form onSubmit={handleSubmitForm}>
                <Stack spacing={3}>
                  <FormControl fullWidth>
                    <InputLabel id="city-select-label">
                      Оберіть місто
                    </InputLabel>
                    <Select
                      labelId="city-select-label"
                      id="city-select"
                      value={selectedCity}
                      label="Оберіть місто"
                      onChange={handleCityChange}
                      disabled={fetchingCities}
                    >
                      {fetchingCities ? (
                        <MenuItem disabled>
                          <Box sx={{ display: "flex", alignItems: "center" }}>
                            <CircularProgress size={20} sx={{ mr: 1 }} />
                            Завантаження міст...
                          </Box>
                        </MenuItem>
                      ) : cities.length > 0 ? (
                        cities.map((city) => (
                          <MenuItem key={city} value={city}>
                            {city}
                          </MenuItem>
                        ))
                      ) : (
                        <MenuItem disabled>Немає доступних міст</MenuItem>
                      )}
                    </Select>
                  </FormControl>

                  <Typography variant="h6">
                    Порогові значення забруднювачів
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Встановіть пороги, при перевищенні яких ви отримаєте
                    сповіщення. Залиште поле порожнім, якщо не бажаєте
                    отримувати сповіщення для цього показника.
                  </Typography>

                  <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                    <TextField
                      label="PM2.5 (мкг/м³)"
                      value={thresholdPM25}
                      onChange={(e) => setThresholdPM25(e.target.value)}
                      type="number"
                      fullWidth
                    />
                    <TextField
                      label="PM10 (мкг/м³)"
                      value={thresholdPM10}
                      onChange={(e) => setThresholdPM10(e.target.value)}
                      type="number"
                      fullWidth
                    />
                  </Stack>

                  <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                    <TextField
                      label="O3 (озон, мкг/м³)"
                      value={thresholdO3}
                      onChange={(e) => setThresholdO3(e.target.value)}
                      type="number"
                      fullWidth
                    />
                    <TextField
                      label="NO2 (діоксид азоту, мкг/м³)"
                      value={thresholdNO2}
                      onChange={(e) => setThresholdNO2(e.target.value)}
                      type="number"
                      fullWidth
                    />
                  </Stack>

                  <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                    <TextField
                      label="SO2 (діоксид сірки, мкг/м³)"
                      value={thresholdSO2}
                      onChange={(e) => setThresholdSO2(e.target.value)}
                      type="number"
                      fullWidth
                    />
                    <TextField
                      label="CO (монооксид вуглецю, мкг/м³)"
                      value={thresholdCO}
                      onChange={(e) => setThresholdCO(e.target.value)}
                      type="number"
                      fullWidth
                    />
                  </Stack>

                  <TextField
                    label="Індекс якості повітря (AQI)"
                    value={thresholdAQI}
                    onChange={(e) => setThresholdAQI(e.target.value)}
                    type="number"
                    fullWidth
                  />

                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={dailySummary}
                        onChange={(e) => setDailySummary(e.target.checked)}
                      />
                    }
                    label="Отримувати щоденний звіт про якість повітря"
                  />

                  <Stack direction="row" spacing={2} justifyContent="flex-end">
                    <Button
                      variant="outlined"
                      onClick={() => setShowForm(false)}
                    >
                      Скасувати
                    </Button>
                    <Button
                      type="submit"
                      variant="contained"
                      disabled={loading || !selectedCity}
                    >
                      {loading ? (
                        <Box sx={{ display: "flex", alignItems: "center" }}>
                          <CircularProgress
                            size={20}
                            sx={{ mr: 1 }}
                            color="inherit"
                          />
                          Зберігання...
                        </Box>
                      ) : (
                        "Зберегти"
                      )}
                    </Button>
                  </Stack>
                </Stack>
              </form>
            </Paper>
          )}

          {history.length > 0 && (
            <Paper elevation={3} sx={{ p: 4, width: "100%" }}>
              <Typography variant="h5" gutterBottom>
                Історія сповіщень
              </Typography>

              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Дата і час</TableCell>
                      <TableCell>Місто</TableCell>
                      <TableCell>Тип сповіщення</TableCell>
                      <TableCell>Показник</TableCell>
                      <TableCell>Значення</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {history.map((item) => (
                      <TableRow key={item.id}>
                        <TableCell>
                          {new Date(item.sent_at).toLocaleString()}
                        </TableCell>
                        <TableCell>{item.city}</TableCell>
                        <TableCell>
                          {item.notification_type === "threshold"
                            ? "Порогове сповіщення"
                            : "Щоденний звіт"}
                        </TableCell>
                        <TableCell>{item.pollutant || "-"}</TableCell>
                        <TableCell>
                          {item.value ? item.value.toFixed(1) : "-"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          )}
        </>
      )}

      {/* Діалог підтвердження видалення */}
      <Dialog open={deleteDialogOpen} onClose={handleCancelDelete}>
        <DialogTitle>Підтвердження видалення</DialogTitle>
        <DialogContent>
          <Typography>
            Ви дійсно бажаєте видалити це сповіщення? Цю дію неможливо
            скасувати.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancelDelete}>Скасувати</Button>
          <Button onClick={handleDeleteNotification} color="error">
            Видалити
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Notifications;
