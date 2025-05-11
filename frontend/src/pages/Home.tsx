import { useState, useEffect } from "react";
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
  SelectChangeEvent,
  CircularProgress,
  Alert,
} from "@mui/material";
import { api, AirQualityData } from "../services/api";

// Create interfaces for the Home page
interface AQIForecast {
  city: string;
  date: string;
  aqi: number;
  pollutants: {
    pm2_5: number;
    pm10: number;
    o3: number;
    no2: number;
    so2: number;
    co: number;
  };
  forecast: {
    day: string;
    aqi: number;
    quality: string;
  }[];
}

const Home = () => {
  const [cities, setCities] = useState<string[]>([]);
  const [selectedCity, setSelectedCity] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [fetchingCities, setFetchingCities] = useState<boolean>(false);
  const [forecastData, setForecastData] = useState<AQIForecast | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch cities when component mounts
    const fetchCities = async () => {
      setFetchingCities(true);
      setError(null);
      try {
        const citiesData = await api.getCities();
        console.log("Отримані міста:", citiesData);
        setCities(citiesData);
      } catch (error) {
        console.error("Помилка отримання міст:", error);
        setError("Не вдалося завантажити список міст. Спробуйте пізніше.");
      } finally {
        setFetchingCities(false);
      }
    };

    fetchCities();
  }, []);

  const handleCityChange = (event: SelectChangeEvent) => {
    setSelectedCity(event.target.value);
    // Скидаємо попередні дані прогнозу при зміні міста
    setForecastData(null);
  };

  const handleForecastClick = async () => {
    if (!selectedCity) return;

    setLoading(true);
    setError(null);
    try {
      // Get current air quality data
      const airQualityData = await api.getAirQuality(selectedCity);
      // setCurrentData(airQualityData);

      // Get prediction data
      const predictionData = await api.getPrediction(selectedCity, 72); // Get prediction for 72 hours (3 days)

      if (airQualityData.length > 0) {
        // Calculate simple AQI based on latest measurements (this is a simplified method)
        const latest = airQualityData[0];
        const aqi = calculateAQI(latest);

        const getPredictionForDay = (dayOffset: number) => {
          const prediction = predictionData.predictions.find(
            (p) =>
              new Date(p.datetime).getDate() ===
              new Date().getDate() + dayOffset
          );
          return prediction
            ? { ...prediction, timestamp: prediction.datetime }
            : latest;
        };

        // Transform the data to match the expected format
        const forecast: AQIForecast = {
          city: selectedCity,
          date: new Date().toLocaleDateString(),
          aqi: aqi,
          pollutants: {
            pm2_5: latest.pm2_5,
            pm10: latest.pm10,
            o3: latest.o3,
            no2: latest.no2,
            so2: latest.so2,
            co: latest.co,
          },
          forecast: [
            {
              day: "Сьогодні",
              aqi: aqi,
              quality: getAirQualityCategory(aqi),
            },
            {
              day: "Завтра",
              aqi: calculateAQI(getPredictionForDay(1)),
              quality: getAirQualityCategory(
                calculateAQI(getPredictionForDay(1))
              ),
            },
            {
              day: "Післязавтра",
              aqi: calculateAQI(getPredictionForDay(2)),
              quality: getAirQualityCategory(
                calculateAQI(getPredictionForDay(2))
              ),
            },
          ],
        };

        setForecastData(forecast);
      } else {
        setError("Немає даних для вибраного міста");
      }
    } catch (error) {
      console.error("Помилка отримання прогнозу:", error);
      setError("Не вдалося отримати прогноз. Спробуйте пізніше.");
    } finally {
      setLoading(false);
    }
  };

  // Simple calculation of Air Quality Index based on EPA standards (simplified)
  const calculateAQI = (data: AirQualityData): number => {
    if (!data) return 0;

    // Simplified calculation - this should be more complex in a real app
    const pm25Index = Math.round((data.pm2_5 / 12) * 50);
    const pm10Index = Math.round((data.pm10 / 50) * 50);
    const o3Index = Math.round((data.o3 / 100) * 50);
    const no2Index = Math.round((data.no2 / 100) * 50);

    // Return the max of all pollutant indices
    return Math.max(pm25Index, pm10Index, o3Index, no2Index);
  };

  // Get air quality category based on AQI
  const getAirQualityCategory = (aqi: number): string => {
    if (aqi <= 50) return "Добра";
    if (aqi <= 100) return "Помірна";
    if (aqi <= 150) return "Шкідлива для чутливих груп";
    if (aqi <= 200) return "Шкідлива";
    if (aqi <= 300) return "Дуже шкідлива";
    return "Небезпечна";
  };

  // Add recommendations function
  const getRecommendations = (aqi: number): string[] => {
    const recommendations: string[] = [];

    if (aqi <= 50) {
      recommendations.push(
        "Ідеальна погода для активного відпочинку на вулиці"
      );
      recommendations.push("Можна відкривати вікна для провітрювання");
    } else if (aqi <= 100) {
      recommendations.push("Дозволена звичайна активність на вулиці");
      recommendations.push(
        "Чутливим групам рекомендується обмежити тривалі прогулянки"
      );
    } else if (aqi <= 150) {
      recommendations.push(
        "Чутливим групам рекомендується уникати тривалих прогулянок"
      );
      recommendations.push("Зменшіть інтенсивність фізичних вправ на вулиці");
      recommendations.push(
        "Використовуйте кондиціонер або очищувач повітря в приміщенні"
      );
    } else if (aqi <= 200) {
      recommendations.push(
        "Всі групи населення повинні обмежити перебування на вулиці"
      );
      recommendations.push(
        "Уникайте інтенсивних фізичних вправ на відкритому повітрі"
      );
      recommendations.push("Використовуйте засоби захисту органів дихання");
    } else if (aqi <= 300) {
      recommendations.push("Уникайте будь-якої активності на вулиці");
      recommendations.push("Закрийте вікна та використовуйте очищувач повітря");
      recommendations.push("Носіть маску при необхідності виходу на вулицю");
    } else {
      recommendations.push("Уникайте будь-якого перебування на вулиці");
      recommendations.push("Використовуйте очищувач повітря в приміщенні");
      recommendations.push("Носіть маску при необхідності виходу на вулицю");
      recommendations.push(
        "Розгляньте можливість евакуації з зони забруднення"
      );
    }

    return recommendations;
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
      {error && (
        <Alert severity="error" sx={{ mb: 2, width: "100%" }}>
          {error}
        </Alert>
      )}

      <Paper elevation={3} sx={{ p: 4, mb: 4, width: "100%" }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Прогноз Якості Повітря
        </Typography>

        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={2}
          alignItems={{ xs: "stretch", md: "center" }}
          width="100%"
        >
          <Box sx={{ width: "100%" }}>
            <FormControl fullWidth>
              <InputLabel id="city-select-label">Оберіть місто</InputLabel>
              <Select
                labelId="city-select-label"
                id="city-select"
                value={selectedCity}
                label="Оберіть місто"
                onChange={handleCityChange}
                disabled={fetchingCities}
                sx={{ height: "56px" }}
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
          </Box>
          <Box sx={{ width: { xs: "100%", md: "50%" } }}>
            <Button
              variant="contained"
              onClick={handleForecastClick}
              disabled={!selectedCity || loading}
              fullWidth
              sx={{
                height: "56px",
                textTransform: "uppercase",
              }}
            >
              {loading ? (
                <Box sx={{ display: "flex", alignItems: "center" }}>
                  <CircularProgress size={20} sx={{ mr: 1 }} color="inherit" />
                  Завантаження...
                </Box>
              ) : (
                "Отримати прогноз"
              )}
            </Button>
          </Box>
        </Stack>
      </Paper>

      {forecastData && (
        <Paper elevation={3} sx={{ p: 4, width: "100%" }}>
          <Typography variant="h5" gutterBottom>
            Прогноз якості повітря для міста {forecastData.city}
          </Typography>

          <Stack spacing={3} sx={{ width: "100%" }}>
            <Box width="100%">
              <Paper
                sx={{ p: 2, bgcolor: "background.default", width: "100%" }}
              >
                <Typography variant="h6" gutterBottom>
                  Поточний індекс якості повітря (AQI): {forecastData.aqi}
                </Typography>
                <Typography variant="body1">
                  Дата: {forecastData.date}
                </Typography>
              </Paper>
            </Box>

            {/* Add recommendations section */}
            <Box width="100%">
              <Typography variant="h6" gutterBottom>
                Рекомендації
              </Typography>
              <Paper sx={{ p: 2, bgcolor: "background.default" }}>
                <Stack spacing={1}>
                  {getRecommendations(forecastData.aqi).map(
                    (recommendation, index) => (
                      <Typography key={index} variant="body1">
                        • {recommendation}
                      </Typography>
                    )
                  )}
                </Stack>
              </Paper>
            </Box>

            <Box width="100%">
              <Typography variant="h6" gutterBottom>
                Рівні забруднювачів
              </Typography>
              <Stack direction="row" flexWrap="wrap" spacing={2} useFlexGap>
                {Object.entries(forecastData.pollutants).map(([key, value]) => (
                  <Box
                    key={key}
                    sx={{
                      width: {
                        xs: "calc(50% - 8px)",
                        sm: "calc(33% - 8px)",
                        md: "calc(16.66% - 8px)",
                      },
                      minWidth: "100px",
                    }}
                  >
                    <Paper sx={{ p: 2, textAlign: "center", height: "100%" }}>
                      <Typography variant="body2" color="text.secondary">
                        {key.toUpperCase()}
                      </Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {value.toFixed(1)}
                      </Typography>
                    </Paper>
                  </Box>
                ))}
              </Stack>
            </Box>

            <Box width="100%">
              <Typography variant="h6" gutterBottom>
                Прогноз на 3 дні
              </Typography>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                {forecastData.forecast.map((day, index) => (
                  <Box key={index} sx={{ width: { xs: "100%", sm: "33.33%" } }}>
                    <Paper sx={{ p: 2, textAlign: "center", height: "100%" }}>
                      <Typography variant="body1" fontWeight="bold">
                        {day.day}
                      </Typography>
                      <Typography variant="h6" color="primary">
                        AQI: {day.aqi}
                      </Typography>
                      <Typography variant="body2">{day.quality}</Typography>
                    </Paper>
                  </Box>
                ))}
              </Stack>
            </Box>
          </Stack>
        </Paper>
      )}
    </Box>
  );
};

export default Home;
