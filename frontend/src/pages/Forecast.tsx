import { useState, useEffect } from "react";
import {
  Typography,
  Box,
  Paper,
  Grid,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  SelectChangeEvent,
} from "@mui/material";
import { api, AirQualityData, PredictionResult } from "../services/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import ModelMetrics from "../components/ModelMetrics";

const Forecast = () => {
  const [cities, setCities] = useState<string[]>([]);
  const [selectedCity, setSelectedCity] = useState<string>("");
  const [currentData, setCurrentData] = useState<AirQualityData[]>([]);
  const [predictionData, setPredictionData] = useState<PredictionResult | null>(
    null
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [selectedParameter, setSelectedParameter] = useState<string | null>(
    null
  );

  // Завантаження списку міст при монтуванні компонента
  useEffect(() => {
    const loadCities = async () => {
      const cityList = await api.getCities();
      setCities(cityList);
      if (cityList.length > 0) {
        setSelectedCity(cityList[0]);
      }
    };

    loadCities();
  }, []);

  // Завантаження даних про якість повітря при зміні вибраного міста
  useEffect(() => {
    if (selectedCity) {
      fetchAirQualityData(selectedCity);
    }
  }, [selectedCity]);

  const fetchAirQualityData = async (city: string) => {
    setLoading(true);
    try {
      // Отримати поточні дані
      const airQualityData = await api.getAirQuality(city);

      setCurrentData(airQualityData);

      // Отримати прогноз
      const prediction = await api.getPrediction(city, 72);
      setPredictionData(prediction);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCityChange = (event: SelectChangeEvent) => {
    setSelectedCity(event.target.value);
  };

  const handleParameterClick = (parameter: string) => {
    setSelectedParameter(parameter);
  };

  // Форматувати дані для графіків
  const formatDataForChart = (data: AirQualityData[]) => {
    return data.map((item) => ({
      date: new Date(item.timestamp).toLocaleString(),
      PM2_5: item.pm2_5,
      PM10: item.pm10,
      O3: item.o3,
      NO2: item.no2,
      SO2: item.so2,
      CO: item.co,
    }));
  };

  // Отримати останні дані про якість повітря
  const getLatestData = () => {
    if (currentData.length === 0) return null;
    return currentData[0];
  };

  const latestData = getLatestData();

  const renderParameterCard = (
    label: string,
    value: number,
    parameter: string
  ) => (
    <Grid size={{ xs: 6, sm: 4, md: 2 }}>
      <Paper
        sx={{
          p: 2,
          textAlign: "center",
          cursor: "pointer",
          transition: "all 0.3s ease",
          "&:hover": {
            backgroundColor: "action.hover",
            transform: "scale(1.02)",
          },
          backgroundColor:
            selectedParameter === parameter
              ? "action.selected"
              : "background.paper",
        }}
        onClick={() => handleParameterClick(parameter)}
      >
        <Typography variant="subtitle2">{label}</Typography>
        <Typography variant="h6">{value.toFixed(1)}</Typography>
      </Paper>
    </Grid>
  );

  const renderParameterChart = () => {
    if (!selectedParameter || !predictionData) return null;

    const data = formatDataForChart(
      predictionData.predictions.map((p) => ({
        ...p,
        timestamp: p.datetime,
      }))
    );
    const colors = {
      PM2_5: "#8884d8",
      PM10: "#82ca9d",
      O3: "#ffc658",
      NO2: "#ff8042",
      SO2: "#0088fe",
      CO: "#00C49F",
    };

    return (
      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography variant="h6" gutterBottom>
          Прогноз для {selectedParameter}
        </Typography>
        <Box sx={{ height: 300, width: "100%" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey={selectedParameter}
                stroke={colors[selectedParameter as keyof typeof colors]}
                name={selectedParameter}
              />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      </Paper>
    );
  };

  return (
    <Box sx={{ mt: 4, width: "100%", maxWidth: "100%" }}>
      <Paper elevation={3} sx={{ p: 4, width: "100%" }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 3,
            width: "100%",
            flexDirection: { xs: "column", md: "row" },
            gap: { xs: 2, md: 0 },
          }}
        >
          <Typography variant="h4" component="h1">
            Прогноз якості повітря
          </Typography>

          <FormControl sx={{ minWidth: { xs: "100%", md: 200 } }}>
            <InputLabel id="city-select-label">Місто</InputLabel>
            <Select
              labelId="city-select-label"
              value={selectedCity}
              label="Місто"
              onChange={handleCityChange}
              fullWidth
            >
              {cities.map((city) => (
                <MenuItem key={city} value={city}>
                  {city}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        {loading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 5 }}>
            <CircularProgress />
          </Box>
        ) : (
          <Grid container spacing={3}>
            <Grid size={{ xs: 12 }}>{/* <ModelMetrics /> */}</Grid>

            <Grid size={{ xs: 12 }}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Поточна якість повітря в {selectedCity}
                </Typography>
                {latestData ? (
                  <Box sx={{ mt: 2, width: "100%" }}>
                    <Typography variant="body1">
                      Дата вимірювання:{" "}
                      {new Date(latestData.timestamp).toLocaleString()}
                    </Typography>
                    <Grid container spacing={2} sx={{ mt: 1 }}>
                      {renderParameterCard("PM2.5", latestData.pm2_5, "PM2_5")}
                      {renderParameterCard("PM10", latestData.pm10, "PM10")}
                      {renderParameterCard("O3", latestData.o3, "O3")}
                      {renderParameterCard("NO2", latestData.no2, "NO2")}
                      {renderParameterCard("SO2", latestData.so2, "SO2")}
                      {renderParameterCard("CO", latestData.co, "CO")}
                    </Grid>
                    {renderParameterChart()}
                  </Box>
                ) : (
                  <Typography variant="body1">
                    Немає доступних даних для відображення.
                  </Typography>
                )}
              </Paper>
            </Grid>
          </Grid>
        )}
      </Paper>
    </Box>
  );
};

export default Forecast;
