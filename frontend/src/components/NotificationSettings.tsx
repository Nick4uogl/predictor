import { useState } from "react";
import {
  Box,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
} from "@mui/icons-material";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";

interface NotificationSettings {
  id?: number;
  email: string;
  city_name: string;
  is_enabled: boolean;
  notification_frequency: string;
  email_notifications: boolean;
  push_notifications: boolean;
  threshold_pm25: number;
  threshold_pm10: number;
  threshold_o3: number;
  threshold_no2: number;
  threshold_so2: number;
  threshold_co: number;
  threshold_aqi: number;
  daily_summary: boolean;
  is_active: boolean;
}

interface NotificationSettingsProps {
  notifications: NotificationSettings[];
  onUpdate: () => void;
}

const NotificationSettings = ({
  notifications,
  onUpdate,
}: NotificationSettingsProps) => {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const [editingNotification, setEditingNotification] =
    useState<NotificationSettings | null>(null);
  const [formData, setFormData] = useState<NotificationSettings>({
    email: user?.email || "",
    city_name: "",
    is_enabled: true,
    notification_frequency: "daily",
    email_notifications: true,
    push_notifications: true,
    threshold_pm25: 25.0,
    threshold_pm10: 50.0,
    threshold_o3: 100.0,
    threshold_no2: 40.0,
    threshold_so2: 20.0,
    threshold_co: 7000.0,
    threshold_aqi: 100.0,
    daily_summary: true,
    is_active: true,
  });

  const handleOpen = (notification: NotificationSettings | null = null) => {
    if (notification) {
      setEditingNotification(notification);
      setFormData(notification);
    } else {
      setEditingNotification(null);
      setFormData({
        email: user?.email || "",
        city_name: "",
        is_enabled: true,
        notification_frequency: "daily",
        email_notifications: true,
        push_notifications: true,
        threshold_pm25: 25.0,
        threshold_pm10: 50.0,
        threshold_o3: 100.0,
        threshold_no2: 40.0,
        threshold_so2: 20.0,
        threshold_co: 7000.0,
        threshold_aqi: 100.0,
        daily_summary: true,
        is_active: true,
      });
    }
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setEditingNotification(null);
  };

  const handleSubmit = async () => {
    try {
      if (editingNotification?.id) {
        await api.updateNotification(editingNotification.id, formData);
      } else {
        await api.createNotification(formData);
      }
      onUpdate();
      handleClose();
    } catch (error) {
      console.error("Error saving notification:", error);
    }
  };

  const handleDelete = async (id: number | undefined) => {
    if (!id) return;
    try {
      await api.deleteNotification(id);
      onUpdate();
    } catch (error) {
      console.error("Error deleting notification:", error);
    }
  };

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 2 }}>
        <Typography variant="h6">Налаштування сповіщень</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => handleOpen()}
        >
          Додати сповіщення
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Місто</TableCell>
              <TableCell>Частота</TableCell>
              <TableCell>Email</TableCell>
              <TableCell>Push</TableCell>
              <TableCell>Дії</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {notifications.map((notification) => (
              <TableRow key={notification.id}>
                <TableCell>{notification.city_name}</TableCell>
                <TableCell>{notification.notification_frequency}</TableCell>
                <TableCell>
                  <Switch checked={notification.email_notifications} disabled />
                </TableCell>
                <TableCell>
                  <Switch checked={notification.push_notifications} disabled />
                </TableCell>
                <TableCell>
                  <IconButton onClick={() => handleOpen(notification)}>
                    <EditIcon />
                  </IconButton>
                  <IconButton onClick={() => handleDelete(notification.id)}>
                    <DeleteIcon />
                  </IconButton>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>
          {editingNotification ? "Редагувати сповіщення" : "Додати сповіщення"}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Місто</InputLabel>
              <Select
                value={formData.city_name}
                onChange={(e) =>
                  setFormData({ ...formData, city_name: e.target.value })
                }
                label="Місто"
              >
                <MenuItem value="Kyiv">Київ</MenuItem>
                <MenuItem value="Lviv">Львів</MenuItem>
                <MenuItem value="Kharkiv">Харків</MenuItem>
                <MenuItem value="Odesa">Одеса</MenuItem>
                <MenuItem value="Dnipro">Дніпро</MenuItem>
              </Select>
            </FormControl>

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Частота сповіщень</InputLabel>
              <Select
                value={formData.notification_frequency}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    notification_frequency: e.target.value,
                  })
                }
                label="Частота сповіщень"
              >
                <MenuItem value="realtime">В реальному часі</MenuItem>
                <MenuItem value="hourly">Щогодини</MenuItem>
                <MenuItem value="daily">Щодня</MenuItem>
              </Select>
            </FormControl>

            <FormControlLabel
              control={
                <Switch
                  checked={formData.email_notifications}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      email_notifications: e.target.checked,
                    })
                  }
                />
              }
              label="Email сповіщення"
              sx={{ mb: 2 }}
            />

            <Typography variant="subtitle1" sx={{ mb: 2 }}>
              Порогові значення
            </Typography>

            <TextField
              fullWidth
              type="number"
              label="PM2.5 (мкг/м³)"
              value={formData.threshold_pm25}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  threshold_pm25: parseFloat(e.target.value),
                })
              }
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              type="number"
              label="PM10 (мкг/м³)"
              value={formData.threshold_pm10}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  threshold_pm10: parseFloat(e.target.value),
                })
              }
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              type="number"
              label="O3 (мкг/м³)"
              value={formData.threshold_o3}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  threshold_o3: parseFloat(e.target.value),
                })
              }
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              type="number"
              label="NO2 (мкг/м³)"
              value={formData.threshold_no2}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  threshold_no2: parseFloat(e.target.value),
                })
              }
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              type="number"
              label="SO2 (мкг/м³)"
              value={formData.threshold_so2}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  threshold_so2: parseFloat(e.target.value),
                })
              }
              sx={{ mb: 2 }}
            />

            <TextField
              fullWidth
              type="number"
              label="CO (мкг/м³)"
              value={formData.threshold_co}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  threshold_co: parseFloat(e.target.value),
                })
              }
              sx={{ mb: 2 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Скасувати</Button>
          <Button onClick={handleSubmit} variant="contained">
            Зберегти
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default NotificationSettings;
