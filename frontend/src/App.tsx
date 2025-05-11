import { ThemeProvider, CssBaseline, Box } from "@mui/material";
import { createTheme } from "@mui/material/styles";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Forecast from "./pages/Forecast";
import About from "./pages/About";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Profile from "./pages/Profile";
import Navigation from "./components/Navigation";
import { AuthProvider } from "./context/AuthContext";

// Create a theme instance
const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#4c76cc", // Синій колір з скріншота
    },
    secondary: {
      main: "#dc004e",
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <AuthProvider>
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              minHeight: "100vh",
              bgcolor: "#f5f5f5", // Світло-сірий фон
              width: "100vw", // Повна ширина вікна
              maxWidth: "100%", // Обмежуємо максимальну ширину
              boxSizing: "border-box", // Включаємо padding в обчислення ширини
              m: 0,
              p: 0,
              overflow: "hidden", // Запобігаємо горизонтальній прокрутці
            }}
          >
            <Navigation />
            <Box
              component="main"
              sx={{
                flex: 1,
                width: "100%",
                p: 0,
              }}
            >
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/forecast" element={<Forecast />} />
                <Route path="/about" element={<About />} />
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />
                <Route path="/profile" element={<Profile />} />
              </Routes>
            </Box>
          </Box>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;
