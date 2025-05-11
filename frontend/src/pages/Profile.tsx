import {
  Box,
  Container,
  Typography,
  Paper,
  Button,
  Tabs,
  Tab,
} from "@mui/material";
import { useAuth } from "../context/AuthContext";
import { useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";
import { api } from "../services/api";
import NotificationSettings from "../components/NotificationSettings";
import NotificationHistory from "../components/NotificationHistory";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`profile-tabpanel-${index}`}
      aria-labelledby={`profile-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Profile = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [tabValue, setTabValue] = useState(0);
  const [notifications, setNotifications] = useState([]);
  const [notificationHistory, setNotificationHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      loadNotifications();
      loadNotificationHistory();
    }
  }, [user]);

  const loadNotifications = async () => {
    try {
      const data = await api.getUserNotifications();
      setNotifications(data);
    } catch (error) {
      console.error("Error loading notifications:", error);
    }
  };

  const loadNotificationHistory = async () => {
    try {
      const data = await api.getUserNotificationHistory();
      setNotificationHistory(data);
    } catch (error) {
      console.error("Error loading notification history:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleLogout = () => {
    navigate("/");
    logout();
  };

  if (!user) {
    navigate("/");
    return null;
  }

  return (
    <Container maxWidth="md">
      <Box
        sx={{
          marginTop: 8,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
        }}
      >
        <Paper
          elevation={3}
          sx={{
            padding: 4,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            width: "100%",
          }}
        >
          <Typography component="h1" variant="h5">
            Профіль користувача
          </Typography>
          <Box sx={{ mt: 3, width: "100%" }}>
            <Typography variant="body1" sx={{ mb: 2 }}>
              <strong>Email:</strong> {user.email}
            </Typography>
            {user.name && (
              <Typography variant="body1" sx={{ mb: 2 }}>
                <strong>Ім'я:</strong> {user.name}
              </Typography>
            )}
          </Box>

          <Box sx={{ width: "100%", mt: 3 }}>
            <Tabs value={tabValue} onChange={handleTabChange} centered>
              <Tab label="Налаштування сповіщень" />
              <Tab label="Історія сповіщень" />
            </Tabs>

            <TabPanel value={tabValue} index={0}>
              <NotificationSettings
                notifications={notifications}
                onUpdate={loadNotifications}
              />
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <NotificationHistory
                history={notificationHistory}
                loading={loading}
              />
            </TabPanel>
          </Box>

          <Button
            variant="contained"
            color="primary"
            onClick={handleLogout}
            sx={{ mt: 3 }}
          >
            Вийти
          </Button>
        </Paper>
      </Box>
    </Container>
  );
};

export default Profile;
