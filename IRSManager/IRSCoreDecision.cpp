#include "IRSCoreDecision.h"

#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits.h>
#include <sstream>
#include <unistd.h>
#include <utility>

#include "../BrainDeepLearn/Interface.h"

class IRSGoalPoints
{
public:
    static IRSThreadTools::ThreadSafeQueue<std::vector<Eigen::Vector3d>>& GetGoalPointsQueue();
};

namespace IRSCoreDecision
{
    class JsonTextParser
    {
    public:
        static std::string ResolveDefaultKeyboardPosePath()
        {
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) == nullptr)
            {
                return "Configure/Keyboard_104_KeyPose.json";
            }
            return std::string(cwd) + "/Configure/Keyboard_104_KeyPose.json";
        }

        static std::string LoadTextFile(std::string filePath)
        {
            std::ifstream input(filePath);
            if (!input)
            {
                return "";
            }

            std::ostringstream buffer;
            buffer << input.rdbuf();
            return buffer.str();
        }

        static bool ExtractNamedObject(std::string text, std::string key, std::string& objectText)
        {
            const std::size_t keyPos = FindQuotedKey(text, key);
            if (keyPos == std::string::npos)
            {
                return false;
            }

            const std::size_t colonPos = text.find(':', keyPos);
            if (colonPos == std::string::npos)
            {
                return false;
            }

            const std::size_t objectStart = text.find('{', colonPos);
            if (objectStart == std::string::npos)
            {
                return false;
            }

            const std::size_t objectEnd = FindMatchingBracket(text, objectStart, '{', '}');
            if (objectEnd == std::string::npos)
            {
                return false;
            }

            objectText = text.substr(objectStart, objectEnd - objectStart + 1);
            return true;
        }

        static bool ExtractNamedArray(std::string text, std::string key, std::string& arrayText)
        {
            const std::size_t keyPos = FindQuotedKey(text, key);
            if (keyPos == std::string::npos)
            {
                return false;
            }

            const std::size_t colonPos = text.find(':', keyPos);
            if (colonPos == std::string::npos)
            {
                return false;
            }

            const std::size_t arrayStart = text.find('[', colonPos);
            if (arrayStart == std::string::npos)
            {
                return false;
            }

            const std::size_t arrayEnd = FindMatchingBracket(text, arrayStart, '[', ']');
            if (arrayEnd == std::string::npos)
            {
                return false;
            }

            arrayText = text.substr(arrayStart, arrayEnd - arrayStart + 1);
            return true;
        }

        static bool ExtractBoolValue(std::string text, std::string key, bool& value)
        {
            const std::size_t keyPos = FindQuotedKey(text, key);
            if (keyPos == std::string::npos)
            {
                return false;
            }

            const std::size_t colonPos = text.find(':', keyPos);
            if (colonPos == std::string::npos)
            {
                return false;
            }

            const std::size_t valuePos = SkipWhitespace(text, colonPos + 1);
            if (text.compare(valuePos, 4, "true") == 0)
            {
                value = true;
                return true;
            }
            if (text.compare(valuePos, 5, "false") == 0)
            {
                value = false;
                return true;
            }
            return false;
        }

        static bool ExtractDoubleValue(std::string text, std::string key, double& value)
        {
            const std::size_t keyPos = FindQuotedKey(text, key);
            if (keyPos == std::string::npos)
            {
                return false;
            }

            const std::size_t colonPos = text.find(':', keyPos);
            if (colonPos == std::string::npos)
            {
                return false;
            }

            std::string numberToken;
            const std::size_t nextPos = ConsumeNumberToken(text, SkipWhitespace(text, colonPos + 1), numberToken);
            if (nextPos == std::string::npos || numberToken.empty())
            {
                return false;
            }

            char* endPtr = nullptr;
            errno = 0;
            const double parsed = std::strtod(numberToken.c_str(), &endPtr);
            if (errno != 0 || endPtr == numberToken.c_str())
            {
                return false;
            }

            value = parsed;
            return true;
        }

        static bool ExtractVector3(std::string text, std::string key, Eigen::Vector3d& value)
        {
            std::string objectText;
            if (!ExtractNamedObject(text, key, objectText))
            {
                return false;
            }

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            const bool ok = ExtractDoubleValue(objectText, "x", x)
                && ExtractDoubleValue(objectText, "y", y)
                && ExtractDoubleValue(objectText, "z", z);
            if (!ok)
            {
                return false;
            }

            value = Eigen::Vector3d(x, y, z);
            return true;
        }

        static bool ExtractQuaternion(std::string text, std::string key, Eigen::Quaterniond& value)
        {
            std::string objectText;
            if (!ExtractNamedObject(text, key, objectText))
            {
                return false;
            }

            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            double w = 1.0;
            const bool ok = ExtractDoubleValue(objectText, "x", x)
                && ExtractDoubleValue(objectText, "y", y)
                && ExtractDoubleValue(objectText, "z", z)
                && ExtractDoubleValue(objectText, "w", w);
            if (!ok)
            {
                return false;
            }

            value = Eigen::Quaterniond(w, x, y, z);
            return true;
        }

        static std::vector<std::string> ParseStringArray(std::string arrayText)
        {
            std::vector<std::string> values;
            std::size_t pos = 0;
            while (pos < arrayText.size())
            {
                pos = SkipWhitespace(arrayText, pos);
                if (pos >= arrayText.size())
                {
                    break;
                }

                if (arrayText[pos] == '"')
                {
                    std::string value;
                    const std::size_t nextPos = ConsumeQuotedString(arrayText, pos, value);
                    if (nextPos == std::string::npos)
                    {
                        break;
                    }
                    values.push_back(value);
                    pos = nextPos;
                    continue;
                }

                ++pos;
            }
            return values;
        }

        static std::vector<std::pair<std::string, std::string>> ParseTopLevelObjectEntries(std::string objectText)
        {
            std::vector<std::pair<std::string, std::string>> entries;
            if (objectText.size() < 2 || objectText.front() != '{' || objectText.back() != '}')
            {
                return entries;
            }

            std::size_t pos = 1;
            while (pos < objectText.size() - 1)
            {
                pos = SkipWhitespace(objectText, pos);
                if (pos >= objectText.size() - 1 || objectText[pos] == '}')
                {
                    break;
                }
                if (objectText[pos] == ',')
                {
                    ++pos;
                    continue;
                }
                if (objectText[pos] != '"')
                {
                    ++pos;
                    continue;
                }

                std::string key;
                const std::size_t nameEnd = ConsumeQuotedString(objectText, pos, key);
                if (nameEnd == std::string::npos)
                {
                    break;
                }

                const std::size_t colonPos = objectText.find(':', nameEnd);
                if (colonPos == std::string::npos)
                {
                    break;
                }

                std::size_t valuePos = SkipWhitespace(objectText, colonPos + 1);
                if (valuePos >= objectText.size())
                {
                    break;
                }

                std::string valueText;
                if (objectText[valuePos] == '{')
                {
                    const std::size_t objectEnd = FindMatchingBracket(objectText, valuePos, '{', '}');
                    if (objectEnd == std::string::npos)
                    {
                        break;
                    }
                    valueText = objectText.substr(valuePos, objectEnd - valuePos + 1);
                    pos = objectEnd + 1;
                }
                else if (objectText[valuePos] == '[')
                {
                    const std::size_t arrayEnd = FindMatchingBracket(objectText, valuePos, '[', ']');
                    if (arrayEnd == std::string::npos)
                    {
                        break;
                    }
                    valueText = objectText.substr(valuePos, arrayEnd - valuePos + 1);
                    pos = arrayEnd + 1;
                }
                else if (objectText[valuePos] == '"')
                {
                    std::string stringValue;
                    const std::size_t stringEnd = ConsumeQuotedString(objectText, valuePos, stringValue);
                    if (stringEnd == std::string::npos)
                    {
                        break;
                    }
                    valueText = objectText.substr(valuePos, stringEnd - valuePos);
                    pos = stringEnd;
                }
                else
                {
                    std::size_t valueEnd = valuePos;
                    while (valueEnd < objectText.size() && objectText[valueEnd] != ',' && objectText[valueEnd] != '}')
                    {
                        ++valueEnd;
                    }
                    valueText = objectText.substr(valuePos, valueEnd - valuePos);
                    pos = valueEnd;
                }

                entries.emplace_back(key, valueText);
            }

            return entries;
        }

    private:
        static std::size_t FindQuotedKey(std::string text, std::string key)
        {
            const std::string token = std::string("\"") + key + "\"";
            return text.find(token);
        }

        static std::size_t FindMatchingBracket(std::string text, std::size_t startPos, char openChar, char closeChar)
        {
            if (startPos >= text.size() || text[startPos] != openChar)
            {
                return std::string::npos;
            }

            int depth = 0;
            bool inString = false;
            bool escaped = false;
            for (std::size_t i = startPos; i < text.size(); ++i)
            {
                const char c = text[i];
                if (inString)
                {
                    if (escaped)
                    {
                        escaped = false;
                    }
                    else if (c == '\\')
                    {
                        escaped = true;
                    }
                    else if (c == '"')
                    {
                        inString = false;
                    }
                    continue;
                }

                if (c == '"')
                {
                    inString = true;
                    continue;
                }
                if (c == openChar)
                {
                    ++depth;
                    continue;
                }
                if (c == closeChar)
                {
                    --depth;
                    if (depth == 0)
                    {
                        return i;
                    }
                }
            }
            return std::string::npos;
        }

        static std::size_t SkipWhitespace(std::string text, std::size_t pos)
        {
            while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            {
                ++pos;
            }
            return pos;
        }

        static std::size_t ConsumeQuotedString(std::string text, std::size_t pos, std::string& value)
        {
            if (pos >= text.size() || text[pos] != '"')
            {
                return std::string::npos;
            }

            value.clear();
            bool escaped = false;
            for (std::size_t i = pos + 1; i < text.size(); ++i)
            {
                const char c = text[i];
                if (escaped)
                {
                    value.push_back(c);
                    escaped = false;
                    continue;
                }
                if (c == '\\')
                {
                    escaped = true;
                    continue;
                }
                if (c == '"')
                {
                    return i + 1;
                }
                value.push_back(c);
            }
            return std::string::npos;
        }

        static std::size_t ConsumeNumberToken(std::string text, std::size_t pos, std::string& value)
        {
            value.clear();
            if (pos >= text.size())
            {
                return std::string::npos;
            }

            std::size_t i = pos;
            if (text[i] == '+' || text[i] == '-')
            {
                value.push_back(text[i]);
                ++i;
            }

            while (i < text.size())
            {
                const char c = text[i];
                if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
                {
                    value.push_back(c);
                    ++i;
                    continue;
                }
                break;
            }

            return value.empty() ? std::string::npos : i;
        }
    };

    Keyboard104PoseStore::Keyboard104PoseStore()
    {
    }

    bool Keyboard104PoseStore::Load(std::string filePath)
    {
        if (filePath.empty())
        {
            filePath = JsonTextParser::ResolveDefaultKeyboardPosePath();
        }

        const std::string text = JsonTextParser::LoadTextFile(filePath);
        if (text.empty())
        {
            IRS_MESSAGE("Keyboard104PoseStore failed to load file: %s", filePath.c_str());
            return false;
        }

        std::string keyboardPoseObject;
        std::string keysObject;
        if (!JsonTextParser::ExtractNamedObject(text, "keyboard_pose", keyboardPoseObject)
            || !JsonTextParser::ExtractNamedObject(text, "keys", keysObject))
        {
            IRS_MESSAGE("Keyboard104PoseStore failed to parse keyboard pose root fields");
            return false;
        }

        KeyboardPoseData keyboardPose;
        if (!JsonTextParser::ExtractVector3(keyboardPoseObject, "position", keyboardPose.position))
        {
            IRS_MESSAGE("Keyboard104PoseStore failed to parse keyboard position");
            return false;
        }
        if (!JsonTextParser::ExtractQuaternion(keyboardPoseObject, "rotation", keyboardPose.rotation))
        {
            IRS_MESSAGE("Keyboard104PoseStore failed to parse keyboard rotation");
            return false;
        }
        if (keyboardPose.rotation.norm() <= 1e-8)
        {
            keyboardPose.rotation = Eigen::Quaterniond::Identity();
        }
        else
        {
            keyboardPose.rotation.normalize();
        }

        std::map<std::string, KeyPoseData> keyPoseMap;
        const std::vector<std::pair<std::string, std::string>> keyEntries = JsonTextParser::ParseTopLevelObjectEntries(keysObject);
        for (const auto& keyEntry : keyEntries)
        {
            const std::string& keyName = keyEntry.first;
            const std::string& keyObject = keyEntry.second;
            KeyPoseData keyPose;

            if (!JsonTextParser::ExtractBoolValue(keyObject, "present", keyPose.present))
            {
                keyPose.present = false;
            }

            if (keyPose.present && JsonTextParser::ExtractVector3(keyObject, "local_press_point", keyPose.localPressPoint))
            {
                keyPose.worldPressPoint = keyboardPose.position + keyboardPose.rotation * keyPose.localPressPoint;
            }

            keyPoseMap[keyName] = keyPose;
        }

        std::lock_guard<std::mutex> lock(dataMutex_);
        loadedFilePath_ = filePath;
        keyboardPose_ = keyboardPose;
        keyPoseMap_ = keyPoseMap;
        return true;
    }

    void Keyboard104PoseStore::Reset()
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        loadedFilePath_.clear();
        keyboardPose_ = KeyboardPoseData();
        keyPoseMap_.clear();
    }

    bool Keyboard104PoseStore::HasKey(std::string keyName)
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        const auto it = keyPoseMap_.find(keyName);
        return it != keyPoseMap_.end() && it->second.present;
    }

    bool Keyboard104PoseStore::GetWorldPressPoint(std::string keyName, Eigen::Vector3d& point)
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        const auto it = keyPoseMap_.find(keyName);
        if (it == keyPoseMap_.end() || !it->second.present)
        {
            return false;
        }

        point = it->second.worldPressPoint;
        return true;
    }

    std::string Keyboard104PoseStore::GetLoadedFilePath()
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        return loadedFilePath_;
    }

    BrainDecisionNode::BrainDecisionNode()
    {
        keyboardPoseStore_.Load();
        RegisterTask([this] { PopDecisionAndPushGoalPoints(); });
    }

    BrainDecisionNode* BrainDecisionNode::GetInstance()
    {
        static BrainDecisionNode* instance = new BrainDecisionNode();
        return instance;
    }

    void BrainDecisionNode::Reset()
    {
        keyboardPoseStore_.Reset();
        keyboardPoseStore_.Load();

        std::lock_guard<std::mutex> lock(dataMutex_);
        lastDecisionJson_.clear();
        lastGoalPoints_.clear();
    }

    void BrainDecisionNode::PopDecisionAndPushGoalPoints()
    {
        const std::string jsonText = BrainDeepLearnInterface::GetJsonQueue().pop();
        if (jsonText.empty())
        {
            return;
        }

        std::vector<Eigen::Vector3d> goalPoints = DecodeDecisionJson(jsonText);
        {
            std::lock_guard<std::mutex> lock(dataMutex_);
            lastDecisionJson_ = jsonText;
            lastGoalPoints_ = goalPoints;
        }

        if (!goalPoints.empty())
        {
            IRSGoalPoints::GetGoalPointsQueue().push(goalPoints);
        }
    }

    std::vector<Eigen::Vector3d> BrainDecisionNode::DecodeDecisionJson(std::string jsonText)
    {
        std::string keyNamesArray;
        if (!JsonTextParser::ExtractNamedArray(jsonText, "key_names", keyNamesArray))
        {
            return {};
        }

        const std::vector<std::string> keyNames = JsonTextParser::ParseStringArray(keyNamesArray);
        std::vector<Eigen::Vector3d> goalPoints;
        for (const std::string& keyName : keyNames)
        {
            Eigen::Vector3d worldPoint;
            if (keyboardPoseStore_.GetWorldPressPoint(keyName, worldPoint))
            {
                goalPoints.push_back(worldPoint);
            }
        }

        return goalPoints;
    }

    std::string BrainDecisionNode::GetLastDecisionJson()
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        return lastDecisionJson_;
    }

    std::vector<Eigen::Vector3d> BrainDecisionNode::GetLastGoalPoints()
    {
        std::lock_guard<std::mutex> lock(dataMutex_);
        return lastGoalPoints_;
    }
}
