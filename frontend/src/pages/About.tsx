import {
  Typography,
  Box,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  Grid,
  Card,
  CardContent,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import FeaturedPlayListIcon from "@mui/icons-material/FeaturedPlayList";
import DevicesIcon from "@mui/icons-material/Devices";

const About = () => {
  return (
    <Box sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Box display="flex" alignItems="center" mb={3}>
          <InfoIcon fontSize="large" color="primary" sx={{ mr: 2 }} />
          <Typography variant="h4" component="h1">
            Про сервіс прогнозування якості повітря
          </Typography>
        </Box>

        <Typography variant="body1" paragraph>
          Цей додаток призначений для надання точних та актуальних прогнозів
          якості повітря, щоб допомогти користувачам приймати обґрунтовані
          рішення щодо своєї діяльності на відкритому повітрі.
        </Typography>

        <Typography variant="body1" paragraph>
          Використовуючи сучасні технології машинного навчання та аналізу даних,
          ми обробляємо дані з різних джерел, щоб передбачити рівні
          забруднювачів повітря та індекс якості повітря (AQI) для різних міст
          України.
        </Typography>

        <Divider sx={{ my: 3 }} />

        <Box mb={4}>
          <Box display="flex" alignItems="center" mb={2}>
            <FeaturedPlayListIcon color="primary" sx={{ mr: 1 }} />
            <Typography variant="h5" gutterBottom>
              Можливості
            </Typography>
          </Box>

          <Grid container spacing={3}>
            <Grid size={{ xs: 12, md: 4 }}>
              <Card sx={{ height: "100%" }}>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Дані в реальному часі
                  </Typography>
                  <Typography variant="body2">
                    Отримуйте актуальну інформацію про якість повітря для вашого
                    місцезнаходження
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid size={{ xs: 12, md: 4 }}>
              <Card sx={{ height: "100%" }}>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Прогнози
                  </Typography>
                  <Typography variant="body2">
                    Перегляд прогнозів якості повітря на майбутні дні
                  </Typography>
                </CardContent>
              </Card>
            </Grid>

            <Grid size={{ xs: 12, md: 4 }}>
              <Card sx={{ height: "100%" }}>
                <CardContent>
                  <Typography variant="h6" color="primary" gutterBottom>
                    Історичні дані
                  </Typography>
                  <Typography variant="body2">
                    Доступ до історичних даних про якість повітря для аналізу
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 3 }} />

        <Box>
          <Box display="flex" alignItems="center" mb={2}>
            <DevicesIcon color="primary" sx={{ mr: 1 }} />
            <Typography variant="h5" gutterBottom>
              Технології
            </Typography>
          </Box>

          <Typography variant="body1" paragraph>
            Наш додаток використовує передові технології для надання точних
            прогнозів якості повітря:
          </Typography>

          <List>
            <ListItem>
              <ListItemText
                primary="Штучний інтелект"
                secondary="Використання нейронних мереж LSTM для прогнозування часових рядів"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Обробка великих даних"
                secondary="Аналіз великих обсягів даних про забруднювачі повітря з різних джерел"
              />
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Сучасні веб-технології"
                secondary="Розроблено з використанням React, TypeScript, Material UI та Python Flask"
              />
            </ListItem>
          </List>

          <Typography variant="body1" sx={{ mt: 2 }}>
            Система постійно вдосконалюється для підвищення точності прогнозів
            та забезпечення користувачів найбільш актуальною інформацією про
            якість повітря.
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default About;
