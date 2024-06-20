import { LlmModelType, ModelVariableType } from '@/constants/knowledge';
import { Divider, Flex, Form, InputNumber, Select, Slider, Switch } from 'antd';
import camelCase from 'lodash/camelCase';

import { useTranslate } from '@/hooks/commonHooks';
import { useSelectLlmOptionsByModelType } from '@/hooks/llmHooks';
import { useMemo } from 'react';
import styles from './index.less';

interface IProps {
  prefix?: string;
  handleParametersChange(value: ModelVariableType): void;
}

const LlmSettingItems = ({ prefix, handleParametersChange }: IProps) => {
  const { t } = useTranslate('chat');
  const parameterOptions = Object.values(ModelVariableType).map((x) => ({
    label: t(camelCase(x)),
    value: x,
  }));

  const memorizedPrefix = useMemo(() => (prefix ? [prefix] : []), [prefix]);

  const modelOptions = useSelectLlmOptionsByModelType();

  return (
    <>
      <Form.Item
        label={t('model')}
        name="llm_id"
        tooltip={t('modelTip')}
        rules={[{ required: true, message: t('modelMessage') }]}
      >
        <Select options={modelOptions[LlmModelType.Chat]} showSearch />
      </Form.Item>
      <Divider></Divider>
      <Form.Item
        label={t('freedom')}
        name="parameters"
        tooltip={t('freedomTip')}
        initialValue={ModelVariableType.Precise}
      >
        <Select<ModelVariableType>
          options={parameterOptions}
          onChange={handleParametersChange}
        />
      </Form.Item>
      <Form.Item label={t('temperature')} tooltip={t('temperatureTip')}>
        <Flex gap={20} align="center">
          <Form.Item
            name={'temperatureEnabled'}
            valuePropName="checked"
            noStyle
          >
            <Switch size="small" />
          </Form.Item>
          <Form.Item noStyle dependencies={['temperatureEnabled']}>
            {({ getFieldValue }) => {
              const disabled = !getFieldValue('temperatureEnabled');
              return (
                <>
                  <Flex flex={1}>
                    <Form.Item
                      name={[...memorizedPrefix, 'temperature']}
                      noStyle
                    >
                      <Slider
                        className={styles.variableSlider}
                        max={1}
                        step={0.01}
                        disabled={disabled}
                      />
                    </Form.Item>
                  </Flex>
                  <Form.Item name={[...memorizedPrefix, 'temperature']} noStyle>
                    <InputNumber
                      className={styles.sliderInputNumber}
                      max={1}
                      min={0}
                      step={0.01}
                      disabled={disabled}
                    />
                  </Form.Item>
                </>
              );
            }}
          </Form.Item>
        </Flex>
      </Form.Item>
      <Form.Item label={t('topP')} tooltip={t('topPTip')}>
        <Flex gap={20} align="center">
          <Form.Item name={'topPEnabled'} valuePropName="checked" noStyle>
            <Switch size="small" />
          </Form.Item>
          <Form.Item noStyle dependencies={['topPEnabled']}>
            {({ getFieldValue }) => {
              const disabled = !getFieldValue('topPEnabled');
              return (
                <>
                  <Flex flex={1}>
                    <Form.Item name={[...memorizedPrefix, 'top_p']} noStyle>
                      <Slider
                        className={styles.variableSlider}
                        max={1}
                        step={0.01}
                        disabled={disabled}
                      />
                    </Form.Item>
                  </Flex>
                  <Form.Item name={[...memorizedPrefix, 'top_p']} noStyle>
                    <InputNumber
                      className={styles.sliderInputNumber}
                      max={1}
                      min={0}
                      step={0.01}
                      disabled={disabled}
                    />
                  </Form.Item>
                </>
              );
            }}
          </Form.Item>
        </Flex>
      </Form.Item>
      <Form.Item label={t('presencePenalty')} tooltip={t('presencePenaltyTip')}>
        <Flex gap={20} align="center">
          <Form.Item
            name={'presencePenaltyEnabled'}
            valuePropName="checked"
            noStyle
          >
            <Switch size="small" />
          </Form.Item>
          <Form.Item noStyle dependencies={['presencePenaltyEnabled']}>
            {({ getFieldValue }) => {
              const disabled = !getFieldValue('presencePenaltyEnabled');
              return (
                <>
                  <Flex flex={1}>
                    <Form.Item
                      name={[...memorizedPrefix, 'presence_penalty']}
                      noStyle
                    >
                      <Slider
                        className={styles.variableSlider}
                        max={1}
                        step={0.01}
                        disabled={disabled}
                      />
                    </Form.Item>
                  </Flex>
                  <Form.Item
                    name={[...memorizedPrefix, 'presence_penalty']}
                    noStyle
                  >
                    <InputNumber
                      className={styles.sliderInputNumber}
                      max={1}
                      min={0}
                      step={0.01}
                      disabled={disabled}
                    />
                  </Form.Item>
                </>
              );
            }}
          </Form.Item>
        </Flex>
      </Form.Item>
      <Form.Item
        label={t('frequencyPenalty')}
        tooltip={t('frequencyPenaltyTip')}
      >
        <Flex gap={20} align="center">
          <Form.Item
            name={'frequencyPenaltyEnabled'}
            valuePropName="checked"
            noStyle
          >
            <Switch size="small" />
          </Form.Item>
          <Form.Item noStyle dependencies={['frequencyPenaltyEnabled']}>
            {({ getFieldValue }) => {
              const disabled = !getFieldValue('frequencyPenaltyEnabled');
              return (
                <>
                  <Flex flex={1}>
                    <Form.Item
                      name={[...memorizedPrefix, 'frequency_penalty']}
                      noStyle
                    >
                      <Slider
                        className={styles.variableSlider}
                        max={1}
                        step={0.01}
                        disabled={disabled}
                      />
                    </Form.Item>
                  </Flex>
                  <Form.Item
                    name={[...memorizedPrefix, 'frequency_penalty']}
                    noStyle
                  >
                    <InputNumber
                      className={styles.sliderInputNumber}
                      max={1}
                      min={0}
                      step={0.01}
                      disabled={disabled}
                    />
                  </Form.Item>
                </>
              );
            }}
          </Form.Item>
        </Flex>
      </Form.Item>
      <Form.Item label={t('maxTokens')} tooltip={t('maxTokensTip')}>
        <Flex gap={20} align="center">
          <Form.Item name={'maxTokensEnabled'} valuePropName="checked" noStyle>
            <Switch size="small" />
          </Form.Item>
          <Form.Item noStyle dependencies={['maxTokensEnabled']}>
            {({ getFieldValue }) => {
              const disabled = !getFieldValue('maxTokensEnabled');

              return (
                <>
                  <Flex flex={1}>
                    <Form.Item
                      name={[...memorizedPrefix, 'max_tokens']}
                      noStyle
                    >
                      <Slider
                        className={styles.variableSlider}
                        max={2048}
                        disabled={disabled}
                      />
                    </Form.Item>
                  </Flex>
                  <Form.Item name={[...memorizedPrefix, 'max_tokens']} noStyle>
                    <InputNumber
                      disabled={disabled}
                      className={styles.sliderInputNumber}
                      max={2048}
                      min={0}
                    />
                  </Form.Item>
                </>
              );
            }}
          </Form.Item>
        </Flex>
      </Form.Item>
    </>
  );
};

export default LlmSettingItems;
